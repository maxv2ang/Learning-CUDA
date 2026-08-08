// ============================================================================
// NVIDIA (nvcc) / Iluvatar CoreX (clang++) implementation.
// Both backends expose the standard CUDA runtime (cuda_*), so they share this
// single file. All accumulation is done in fp32; fp16 is converted only at the
// kernel boundary (same as PyTorch's math path).
// ============================================================================
#include <vector>
#include <cmath>
#include <cuda_fp16.h>

#include "../tester/utils.h"

// ---------------------------------------------------------------------------
// Type conversion helpers (fp32 <-> fp16)
// ---------------------------------------------------------------------------
template <typename T> struct CudaTypeTraits;

template <> struct CudaTypeTraits<float> {
  static __device__ __forceinline__ float to_float(float x) { return x; }
  static __device__ __forceinline__ float from_float(float x) { return x; }
};
template <> struct CudaTypeTraits<half> {
  static __device__ __forceinline__ float to_float(half x) { return __half2float(x); }
  static __device__ __forceinline__ half from_float(float x) { return __float2half(x); }
};

// ---------------------------------------------------------------------------
// Block reduction helpers (blockDim.x <= 1024)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}
// Full-block reduction; the result is broadcast to every thread.
// `sdata` must be a shared scratch array of at least 32 floats.
__device__ __forceinline__ float block_reduce_max(float v, float* sdata) {
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) sdata[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = (lane < (blockDim.x >> 5)) ? sdata[lane] : -INFINITY;
    v = warp_reduce_max(v);
    if (lane == 0) sdata[0] = v;
  }
  __syncthreads();
  return sdata[0];
}
__device__ __forceinline__ float block_reduce_sum(float v, float* sdata) {
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) sdata[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) sdata[0] = v;
  }
  __syncthreads();
  return sdata[0];
}

// ============================================================================
// Problem 1 : RMSNorm
// ============================================================================
// One block per row; threads cooperate to sum x^2 over the row, then each
// thread writes back a slice of the normalized row.
template <typename T>
__global__ void rmsnorm_kernel(const T* __restrict__ in,
                               const T* __restrict__ w,
                               T* __restrict__ out,
                               size_t hidden_dim, float eps) {
  extern __shared__ float s_red[];   // scratch for the block reduction
  const int tid = threadIdx.x;
  const size_t row = blockIdx.x;
  const T* inrow  = in  + row * hidden_dim;
  T*       outrow = out + row * hidden_dim;

  float sq = 0.0f;
  for (size_t j = tid; j < hidden_dim; j += blockDim.x)
    sq += CudaTypeTraits<T>::to_float(inrow[j]) * CudaTypeTraits<T>::to_float(inrow[j]);

  const float sum = block_reduce_sum(sq, s_red);
  const float rstd = rsqrtf(sum / (float)hidden_dim + eps);  // 1/sqrt(mean(x^2)+eps)

  for (size_t j = tid; j < hidden_dim; j += blockDim.x)
    outrow[j] = CudaTypeTraits<T>::from_float(
        CudaTypeTraits<T>::to_float(inrow[j]) * rstd * CudaTypeTraits<T>::to_float(w[j]));
}

// ============================================================================
// Problem 2 : Flash Attention / scaled dot product attention
// ============================================================================
// FlashAttention-1 style: tiles of S are streamed through a fixed set of
// registers/shared state using the *online softmax* trick, so the full
// [L, S] attention matrix is never materialized.
//
//   O[t, d] = sum_s  softmax_s(q_t . k_s * scale)  * V[s, d]
//
// Grid :  (target_seq_len, batch_size, query_heads)
// Block:  128 threads, one query row t per block.
//
// Role of a thread inside one S-tile:
//   * dot   phase:  thread = source s  ->  computes  q_t . k_s
//   * accum phase:  thread = output dim d ->  registers acc[d] += p_s * V[s, d]
// This way acc lives in shared memory but each slot is written by exactly one
// thread, so no atomic is needed.
template <typename T>
__global__ void flash_attn_kernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ o, int tgt_seq, int src_seq, int q_heads, int kv_heads,
    int head_dim, float scale, bool is_causal) {
  constexpr int THREADS = 128;          // == tile width in S
  const int tid = threadIdx.x;

  const int b  = blockIdx.y;
  const int qh = blockIdx.z;
  const int t  = blockIdx.x;            // target (query) position

  // GQA: query head qh reads KV head (qh / repeat), matching PyTorch's
  // repeat_interleave semantics. Guard against a degenerate kv_heads == 0.
  const int repeat = (kv_heads > 0) ? (q_heads / kv_heads) : 1;
  const int kvh = (repeat > 0) ? (qh / repeat) : qh;

  extern __shared__ float smem[];
  float* s_q   = smem;                            // head_dim  : staged query row
  float* s_acc = smem + head_dim;                 // head_dim  : online-softmax weighted sum
  float* s_p   = smem + 2 * head_dim;             // THREADS   : p_s per source in current tile
  float* s_red = smem + 2 * head_dim + THREADS;   // 32        : reduction scratch

  const long long q_base   = ((long long)b * tgt_seq + t) * q_heads + qh;
  const long long kv_base  = ((long long)b * src_seq) * kv_heads + kvh;

  // Stage the query row and zero the accumulator.
  for (int d = tid; d < head_dim; d += THREADS)
    s_q[d] = CudaTypeTraits<T>::to_float(q[q_base * head_dim + d]);
  for (int d = tid; d < head_dim; d += THREADS)
    s_acc[d] = 0.0f;
  __syncthreads();

  float m = -INFINITY;   // running max of scores (online softmax)
  float l = 0.0f;        // running denominator  sum(exp(score - m))

  for (int s0 = 0; s0 < src_seq; s0 += THREADS) {
    const int s = s0 + tid;
    const bool masked = (s >= src_seq) || (is_causal && s > t);

    // ---- dot phase: this thread owns source s -----------------------------
    float dot = -INFINITY;
    if (!masked) {
      const T* krow = k + (kv_base + (long long)s * kv_heads) * head_dim;
      float acc_dot = 0.0f;
      for (int d = 0; d < head_dim; d++)
        acc_dot += s_q[d] * CudaTypeTraits<T>::to_float(krow[d]);
      dot = acc_dot * scale;
    }

    // Running max over this tile, then combine with the running state.
    const float m_tile = block_reduce_max(dot, s_red);
    if (m_tile != -INFINITY) {           // skip fully-masked tiles
      const float m_new = fmaxf(m, m_tile);
      const float p = masked ? 0.0f : expf(dot - m_new);
      const float l_tile = block_reduce_sum(p, s_red);
      const float corr = expf(m - m_new);      // rescale old accumulator

      s_p[tid] = p;
      __syncthreads();                         // publish s_p[] before accum phase

      // ---- accum phase: this thread owns output dim d ---------------------
      // For each source in this tile, add p_s * V[s, d]; rescale the existing
      // accumulator by corr to account for the moving max m_new.
      for (int d = tid; d < head_dim; d += THREADS) {
        float a = 0.0f;
        for (int si = 0; si < THREADS; si++) {
          const int src = s0 + si;
          if (src < src_seq) {
            const T* vrow = v + (kv_base + (long long)src * kv_heads) * head_dim;
            a += s_p[si] * CudaTypeTraits<T>::to_float(vrow[d]);
          }
        }
        s_acc[d] = s_acc[d] * corr + a;
      }

      m = m_new;
      l = l * corr + l_tile;
      __syncthreads();                         // s_acc[] ready before next tile
    }
  }

  // Normalize by the running denominator and write the output row.
  const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int d = tid; d < head_dim; d += THREADS)
    o[q_base * head_dim + d] = CudaTypeTraits<T>::from_float(s_acc[d] * inv_l);
}

// ============================================================================
// Host wrappers
// ============================================================================
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  const size_t total = rows * hidden_dim;
  if (total == 0) return;

  T *d_in = nullptr, *d_w = nullptr, *d_out = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_in,  total * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_w,   hidden_dim * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_out, total * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_in, h_input.data(),  total * sizeof(T),      cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_w,  h_weight.data(), hidden_dim * sizeof(T), cudaMemcpyHostToDevice));

  const int threads = 256;
  const size_t smem = 32 * sizeof(float);
  rmsnorm_kernel<T><<<(int)rows, threads, smem>>>(d_in, d_w, d_out, hidden_dim, eps);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_out, total * sizeof(T), cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_in));
  RUNTIME_CHECK(cudaFree(d_w));
  RUNTIME_CHECK(cudaFree(d_out));
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  const size_t qn = (size_t)batch_size * target_seq_len * query_heads * head_dim;
  const size_t kn = (size_t)batch_size * src_seq_len  * kv_heads    * head_dim;
  const size_t on = qn;
  if (qn == 0) return;

  T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_q, qn * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, kn * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, kn * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, on * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), qn * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kn * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kn * sizeof(T), cudaMemcpyHostToDevice));

  const float scale = 1.0f / sqrtf((float)head_dim);   // SDPA default scaling
  constexpr int THREADS = 128;
  const dim3 grid(target_seq_len, batch_size, query_heads);
  const size_t smem = (2 * (size_t)head_dim + THREADS + 32) * sizeof(float);

  flash_attn_kernel<T><<<grid, THREADS, smem>>>(
      d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
      head_dim, scale, is_causal);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, on * sizeof(T), cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
