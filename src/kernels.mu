// ============================================================================
// Moore Threads MUSA (mcc) implementation.
// MUSA is CUDA-compatible, so the device kernels below are byte-identical to
// kernels.cu; only the fp16 header and the host runtime (musa_*) differ.
// All accumulation is done in fp32; fp16 is converted at the kernel boundary.
// ============================================================================
#include <vector>
#include <cmath>
#include <musa_fp16.h>

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

// Vector width (in elements) of a 16-byte aligned load for type T.
template <typename T> __host__ __device__ constexpr int kVecW();
template <> __host__ __device__ constexpr int kVecW<float>() { return 4; }
template <> __host__ __device__ constexpr int kVecW<half>()  { return 8; }

// ---------------------------------------------------------------------------
// Warp reduction helpers (result broadcast to every lane via shuffle).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max_bcast(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}
__device__ __forceinline__ float warp_reduce_sum_bcast(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}
// Block reduction helpers (blockDim.x <= 1024), used by rmsNorm.
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

// ---------------------------------------------------------------------------
// Staging of a [BC x head_dim] K/V tile into shared float.
//
// The tile rows (sources) live in global memory strided by `kv_heads*head_dim`
// elements (row-major [batch, src, kv_heads, head_dim] layout), so when
// kv_heads > 1 (GQA) consecutive sources are NOT contiguous. Both paths honor
// that stride. Half is widened to float here so the hot dot/accum loops never
// re-convert.
//
// Vectorized path (`KVStageVec<T>::run`): requires a fully in-bounds tile
// (rows_valid == BC) and a 16-byte aligned, vector-width-divisible head_dim
// (checked by the caller). Loads via float4 (fp32) / uint4 (fp16, 8 halves)
// from global, then writes widened fp32 into the PADDED shared tile.
//
// PADDING (`hs = head_dim+1`): the shared K/V rows are stored with stride hs
// instead of head_dim so that lanes reading their own row at a common column
// (dot phase) hit distinct banks (see kernel). This removes the 32-way shared
// bank conflict when head_dim is a multiple of 32.
// ---------------------------------------------------------------------------
template <typename T> struct KVStageVec;

template <> struct KVStageVec<float> {
  __device__ __forceinline__ static void run(const float* __restrict__ g,
                                             float* __restrict__ s, int head_dim,
                                             int hs, int bc, int kv_heads,
                                             int tid, int nthreads) {
    const int chunks = head_dim >> 2;               // float4 per source row
    const int vec_stride = kv_heads * chunks;       // float4 stride between rows
    const float4* g4 = reinterpret_cast<const float4*>(g);
    for (int idx = tid; idx < bc * chunks; idx += nthreads) {
      const int c = idx / chunks, ch = idx - c * chunks;
      const float4 x = __ldg(g4 + c * vec_stride + ch);
      const int base = c * hs + ch * 4;
      s[base] = x.x; s[base + 1] = x.y; s[base + 2] = x.z; s[base + 3] = x.w;
    }
  }
};
template <> struct KVStageVec<half> {
  __device__ __forceinline__ static void run(const half* __restrict__ g,
                                             float* __restrict__ s, int head_dim,
                                             int hs, int bc, int kv_heads,
                                             int tid, int nthreads) {
    const int chunks = head_dim >> 3;               // uint4 (8 halves) per row
    const int vec_stride = kv_heads * chunks;
    const uint4* g4 = reinterpret_cast<const uint4*>(g);
    for (int idx = tid; idx < bc * chunks; idx += nthreads) {
      const int c = idx / chunks, ch = idx - c * chunks;
      const uint4 u = __ldg(g4 + c * vec_stride + ch);
      const half* hp = reinterpret_cast<const half*>(&u);
      const int base = c * hs + ch * 8;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        s[base + e] = __half2float(hp[e]);
    }
  }
};

// Scalar, bounds-guarded fallback (partial / misaligned tile): reads the first
// `rows_valid` source rows (strided by kv_heads*head_dim), zero-fills the rest.
template <typename T>
__device__ __forceinline__ void stage_scalar(const T* __restrict__ g,
                                             float* __restrict__ s, int head_dim,
                                             int hs, int bc, int rows_valid,
                                             int kv_heads, int tid, int nthreads) {
  for (int i = tid; i < bc * head_dim; i += nthreads) {
    const int c = i / head_dim, d = i - c * head_dim;
    s[c * hs + d] = (c < rows_valid)
                        ? CudaTypeTraits<T>::to_float(g[(long long)c * kv_heads * head_dim + d])
                        : 0.0f;
  }
}

// ============================================================================
// Problem 1 : RMSNorm
// ============================================================================
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
// Tiled FlashAttention-1 (online softmax):
//
//   O[t, d] = sum_s  softmax_s(q_t . k_s * scale)  * V[s, d]
//
// * Each block handles BR query rows for a fixed (batch, query_head); the S
//   dimension is streamed in tiles of BC source columns.
// * K/V tiles are staged into shared memory once and reused by all BR rows
//   (reduces global K/V traffic ~BRx). Q, K, V are widened to fp32 in shared.
// * Grid : (ceil(target_seq_len/BR), batch_size, query_heads)
// * Block: BR*BC threads = BR warps, one query row per warp. This keeps the
//   softmax reduction warp-local (shuffle only, no __syncthreads in the S loop).
// * Accumulation into acc[row][d] is done by the lane owning column d, so each
//   shared slot is written by exactly one thread (no atomics).
// * head_dim is supported up to ~128 (bounded by 48KB shared memory).
template <typename T>
__global__ void flash_attn_kernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ o, int tgt_seq, int src_seq, int q_heads, int kv_heads,
    int head_dim, bool is_causal) {
  constexpr int BR = 8;                 // query rows per block
  constexpr int BC = 32;                // source columns per tile (= warp size)
  constexpr int THREADS = BR * BC;      // 256

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;            // 0..BR-1 -> query row within block
  const int lane = tid & 31;            // 0..BC-1 -> source column within tile

  const int b   = blockIdx.y;
  const int qh  = blockIdx.z;
  const int t0  = blockIdx.x * BR;      // first target row of this block
  const int t   = t0 + warp;            // this warp's query row
  const int H   = head_dim;
  const int HS  = H + 1;                // padded shared row stride (bank-conflict free)
  const bool row_valid = (t < tgt_seq); // this query row exists

  // GQA: query head qh reads KV head (qh / repeat), matching PyTorch's
  // repeat_interleave semantics. Guard against a degenerate kv_heads == 0.
  const int repeat = (kv_heads > 0) ? (q_heads / kv_heads) : 1;
  const int kvh = (repeat > 0) ? (qh / repeat) : qh;

  // SDPA scale, computed in fp32 on device (1.0f/sqrtf). Inputs are in [1,10)
  // so dot products are large (~head_dim*25); a 1-ulp scale rounding would be
  // amplified into a ~1e-5 output error under the tight tolerance.
  const float scale = 1.0f / sqrtf((float)head_dim);

  extern __shared__ float smem[];
  float* s_q = smem;                        // [BR*H]      staged query tile
  float* s_k = s_q + BR * H;                // [BC*HS]     padded staged key tile
  float* s_v = s_k + BC * HS;               // [BC*HS]     padded staged value tile
  float* s_p = s_v + BC * HS;               // [BR*BC]     per-tile normalized probs
  float* s_acc = s_p + BR * BC;             // [BR*H]      output accumulator

  // Causal pruning (C): no source beyond the largest diagonal in this block
  // (t0+BR-1) can be attended, so skip those tiles entirely.
  const int s_limit = is_causal ? min(src_seq, t0 + BR) : src_seq;
  const bool use_vec = (H % kVecW<T>() == 0);

  // ---------- stage Q tile + zero the output accumulator ---------------------
  {
    const int rows_valid_q = min(BR, tgt_seq - t0);
    for (int i = tid; i < BR * H; i += THREADS) {
      const int r = i / H, d = i - r * H;
      s_q[i] = (r < rows_valid_q)
                   ? CudaTypeTraits<T>::to_float(q[((long long)(b * tgt_seq + t0 + r) * q_heads + qh) * H + d])
                   : 0.0f;
    }
    for (int i = tid; i < BR * H; i += THREADS) s_acc[i] = 0.0f;
    __syncthreads();
  }

  if (src_seq <= BC) {
    // ---------- single-tile fast path ----------------------------------------
    // Stage K/V once and softmax+output directly. Bit-identical to the two-pass
    // path (same m, l, p/l, FMA sum) but avoids the second K read + dot
    // recompute for small src_seq.
    const T* ksrc = k + (((long long)b * src_seq + 0) * kv_heads + kvh) * H;
    const T* vsrc = v + (((long long)b * src_seq + 0) * kv_heads + kvh) * H;
    const bool full = (BC <= src_seq);
    const int rows_valid = min(BC, src_seq);
    if (full && use_vec) {
      KVStageVec<T>::run(ksrc, s_k, H, HS, BC, kv_heads, tid, THREADS);
      KVStageVec<T>::run(vsrc, s_v, H, HS, BC, kv_heads, tid, THREADS);
    } else {
      stage_scalar<T>(ksrc, s_k, H, HS, BC, rows_valid, kv_heads, tid, THREADS);
      stage_scalar<T>(vsrc, s_v, H, HS, BC, rows_valid, kv_heads, tid, THREADS);
    }
    __syncthreads();                       // K/V tile ready for all warps

    if (row_valid) {
      const int j = lane;                  // global source index == lane
      const bool masked = (j >= src_seq) || (is_causal && j > t);
      float dot = -INFINITY;
      if (!masked) {
        const float* qrow = s_q + warp * H;
        const float* krow = s_k + lane * HS;
        float acc = 0.0f;
#pragma unroll 4
        for (int d = 0; d < H; d++) acc += qrow[d] * krow[d];  // dot
        dot = acc * scale;
      }
      const float m = warp_reduce_max_bcast(dot);
      if (m != -INFINITY) {                // skip fully-masked rows
        const float p = masked ? 0.0f : expf(dot - m);
        const float l = warp_reduce_sum_bcast(p);
        const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        s_p[warp * BC + lane] = p * inv_l;
        __syncwarp();                      // publish (p/l) within this warp
        for (int d = lane; d < H; d += BC) {
          float a = 0.0f;
          for (int si = 0; si < BC; si++) a += s_p[warp * BC + si] * s_v[si * HS + d];
          o[((long long)(b * tgt_seq + t) * q_heads + qh) * H + d] =
              CudaTypeTraits<T>::from_float(a);
        }
      } else {
        for (int d = lane; d < H; d += BC)
          o[((long long)(b * tgt_seq + t) * q_heads + qh) * H + d] =
              CudaTypeTraits<T>::from_float(0.0f);
      }
    }
  } else {
    float m = -INFINITY;   // running max of scores for this row
    float l = 0.0f;        // running softmax denominator for this row
  
    // ---------- Pass 1: online softmax -> final m, l ---------------------------
    // m and l are streamed over source tiles, so large src_seq never needs the
    // full score list resident in shared memory.
    for (int s0 = 0; s0 < s_limit; s0 += BC) {
      const T* ksrc = k + (((long long)b * src_seq + s0) * kv_heads + kvh) * H;
      const bool full = (s0 + BC <= src_seq);
      const int rows_valid = min(BC, src_seq - s0);
      if (full && use_vec) KVStageVec<T>::run(ksrc, s_k, H, HS, BC, kv_heads, tid, THREADS);
      else                 stage_scalar<T>(ksrc, s_k, H, HS, BC, rows_valid, kv_heads, tid, THREADS);
      __syncthreads();                       // K tile ready for all warps
  
      if (row_valid) {
        const int j = s0 + lane;             // global source index
        const bool masked = (j >= src_seq) || (is_causal && j > t);
        float dot = -INFINITY;
        if (!masked) {
          const float* qrow = s_q + warp * H;
          const float* krow = s_k + lane * HS;
          float acc = 0.0f;
  #pragma unroll 4
          for (int d = 0; d < H; d++) acc += qrow[d] * krow[d];  // dot
          dot = acc * scale;
        }
        const float m_tile = warp_reduce_max_bcast(dot);
        if (m_tile != -INFINITY) {           // skip fully-masked tiles
          const float m_new = fmaxf(m, m_tile);
          const float p = masked ? 0.0f : expf(dot - m_new);
          const float l_tile = warp_reduce_sum_bcast(p);
          const float corr = expf(m - m_new);
          m = m_new;
          l = l * corr + l_tile;
        }
      }
      __syncthreads();                       // K tile state free before next staging
    }
  
    // ---------- Pass 2: o[d] = sum_s (p[s]/l) * v[s][d] ------------------------
    // Uses the final m, l from pass 1. Each p is normalized by 1/l before the V
    // accumulation and accumulated with FMA (a single rounding per term). K/V are
    // re-staged here to recompute the scores; staging and block sync stay outside
    // the per-row branch so all threads take them.
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int s0 = 0; s0 < s_limit; s0 += BC) {
      const T* ksrc = k + (((long long)b * src_seq + s0) * kv_heads + kvh) * H;
      const T* vsrc = v + (((long long)b * src_seq + s0) * kv_heads + kvh) * H;
      const bool full = (s0 + BC <= src_seq);
      const int rows_valid = min(BC, src_seq - s0);
      if (full && use_vec) {
        KVStageVec<T>::run(ksrc, s_k, H, HS, BC, kv_heads, tid, THREADS);
        KVStageVec<T>::run(vsrc, s_v, H, HS, BC, kv_heads, tid, THREADS);
      } else {
        stage_scalar<T>(ksrc, s_k, H, HS, BC, rows_valid, kv_heads, tid, THREADS);
        stage_scalar<T>(vsrc, s_v, H, HS, BC, rows_valid, kv_heads, tid, THREADS);
      }
      __syncthreads();                       // K/V tile ready for all warps
  
      if (row_valid) {
        const int j = s0 + lane;
        const bool masked = (j >= src_seq) || (is_causal && j > t);
        float dot = -INFINITY;
        if (!masked) {
          const float* qrow = s_q + warp * H;
          const float* krow = s_k + lane * HS;
          float acc = 0.0f;
  #pragma unroll 4
          for (int d = 0; d < H; d++) acc += qrow[d] * krow[d];
          dot = acc * scale;
        }
        const float p = masked ? 0.0f : expf(dot - m);
        s_p[warp * BC + lane] = p * inv_l;   // (p/l)
        __syncwarp();                        // publish normalized p within this warp
  
        for (int d = lane; d < H; d += BC) {
          float a = 0.0f;
          for (int si = 0; si < BC; si++) a += s_p[warp * BC + si] * s_v[si * HS + d];
          s_acc[warp * H + d] += a;
        }
      }
      __syncthreads();                       // K/V tile state free before next staging
    }
  
    if (row_valid) {
      for (int d = lane; d < H; d += BC)
        o[((long long)(b * tgt_seq + t) * q_heads + qh) * H + d] =
            CudaTypeTraits<T>::from_float(s_acc[warp * H + d]);
    }
  }   // end multi-tile two-pass path (else)
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
  RUNTIME_CHECK(musaMalloc(&d_in,  total * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_w,   hidden_dim * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_out, total * sizeof(T)));
  RUNTIME_CHECK(musaMemcpy(d_in, h_input.data(),  total * sizeof(T),      musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_w,  h_weight.data(), hidden_dim * sizeof(T), musaMemcpyHostToDevice));

  const int threads = 256;
  const size_t smem = 32 * sizeof(float);
  rmsnorm_kernel<T><<<(int)rows, threads, smem>>>(d_in, d_w, d_out, hidden_dim, eps);
  RUNTIME_CHECK(musaGetLastError());
  RUNTIME_CHECK(musaDeviceSynchronize());

  RUNTIME_CHECK(musaMemcpy(h_output.data(), d_out, total * sizeof(T), musaMemcpyDeviceToHost));
  RUNTIME_CHECK(musaFree(d_in));
  RUNTIME_CHECK(musaFree(d_w));
  RUNTIME_CHECK(musaFree(d_out));
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
  RUNTIME_CHECK(musaMalloc(&d_q, qn * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_k, kn * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_v, kn * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_o, on * sizeof(T)));
  RUNTIME_CHECK(musaMemcpy(d_q, h_q.data(), qn * sizeof(T), musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_k, h_k.data(), kn * sizeof(T), musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_v, h_v.data(), kn * sizeof(T), musaMemcpyHostToDevice));

  constexpr int BR = 8, BC = 32;
  const int HS = head_dim + 1;   // padded shared row stride
  const dim3 grid((target_seq_len + BR - 1) / BR, batch_size, query_heads);
  // s_q + 2*BC*HS (s_k, s_v) + BR*BC (s_p) + BR*head_dim (s_acc).
  const size_t smem = ((size_t)BR * head_dim
                       + 2 * (size_t)BC * HS
                       + (size_t)BR * BC
                       + (size_t)BR * head_dim)
                      * sizeof(float);

  flash_attn_kernel<T><<<grid, BR * BC, smem>>>(
      d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
      head_dim, is_causal);
  RUNTIME_CHECK(musaGetLastError());
  RUNTIME_CHECK(musaDeviceSynchronize());

  RUNTIME_CHECK(musaMemcpy(h_o.data(), d_o, on * sizeof(T), musaMemcpyDeviceToHost));
  RUNTIME_CHECK(musaFree(d_q));
  RUNTIME_CHECK(musaFree(d_k));
  RUNTIME_CHECK(musaFree(d_v));
  RUNTIME_CHECK(musaFree(d_o));
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
