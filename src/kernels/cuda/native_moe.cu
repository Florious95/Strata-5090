// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/ordered_moe_fold.hpp"
#include "strata/kernels/batch_deterministic_reduction.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
// Preserve the pre-PR-2 arithmetic verbatim for the default/OFF profile.
// Keep these separate from the RN helpers: fast-math/FTZ is legacy behavior.
namespace legacy {
// GATED (S26 STRATA_LFUSE): `shared` is the shared expert's unscaled output and sg its raw gate logit per token;
// the row is scaled here as shared_expert_multi did (native_scalar_sigmoid_multi_kernel's expression, then one
// rounded multiply = scale_rows_kernel's `out *= g`), then added as before
template<bool ZADD = false, bool GATED = false>
__global__ void combine(const float* __restrict__ parts, const float* __restrict__ weights,
                        const float* __restrict__ shared, float* __restrict__ output,
                        int64_t n_embd, int k, const float* __restrict__ sg = nullptr) {
    // blockIdx.y = the token of a multi-token launch (0 for the single one)
    const int64_t tk = blockIdx.y;
    parts += tk * k * n_embd; weights += tk * k; if (shared) shared += tk * n_embd; output += tk * n_embd;
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    // ZADD: a part is 0.0f + hit (the zeroed row plus moe_hit_add of the verify window), rounded as that add
    float p0 = parts[col];
    if (ZADD) p0 = 0.0f + p0;
    float sum = p0 * weights[0];
    for (int expert = 1; expert < k; ++expert) {
        float p = parts[int64_t(expert) * n_embd + col];
        if (ZADD) p = 0.0f + p;
        sum += p * weights[expert];
    }
    if constexpr (GATED) {
        // contraction off in this block only: the product is rounded on its own (scale_rows' store), then added
#pragma clang fp contract(off)
        const float g = __fdividef(1.0f, 1.0f + __expf(-sg[tk]));
        const float sh = shared[col] * g;
        sum += sh;
    } else {
        if (shared) sum += shared[col];
    }
    output[col] = sum;
}
__global__ void combine_k10_vec4(const float4* __restrict__ parts4, const float* __restrict__ weights,
                                 const float4* __restrict__ shared4, float4* __restrict__ output4,
                                 int64_t n4) {
    const int64_t tk = blockIdx.y;
    parts4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;
    const float w0 = __ldg(weights + 0);
    const float4 p0 = parts4[c4];
    float4 sum = make_float4(p0.x * w0, p0.y * w0, p0.z * w0, p0.w * w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        const float w = __ldg(weights + expert);
        const float4 p = parts4[int64_t(expert) * n4 + c4];
        // the documented contract spelled out: the first product rounded, then one FMA per expert in order. Left to the
        // compiler's contraction, a build may fuse a different product (gfx1151: the float4 kernel then differed from the
        // scalar one, native_multi_parity)
        sum.x = fmaf(p.x, w, sum.x);
        sum.y = fmaf(p.y, w, sum.y);
        sum.z = fmaf(p.z, w, sum.z);
        sum.w = fmaf(p.w, w, sum.w);
    }
    const float4 sh = shared4[c4];
    sum.x += sh.x;
    sum.y += sh.y;
    sum.z += sh.z;
    sum.w += sh.w;
    output4[c4] = sum;
}
__global__ void combine_k10_vec4_hits(const float4* __restrict__ parts4, const float4* __restrict__ hits4,
                                      const int32_t* __restrict__ dst, const int32_t* __restrict__ count,
                                      const float* __restrict__ weights, const float4* __restrict__ shared4,
                                      float4* __restrict__ output4, int64_t n4) {
    const int64_t tk = blockIdx.y;
    __shared__ int hit_mask;
    if (threadIdx.x == 0) hit_mask = 0;
    __syncthreads();
    const int nh = *count;
    for (int h = threadIdx.x; h < nh; h += blockDim.x) {
        const int r = dst[h];
        if (r / 10 == (int) tk) atomicOr(&hit_mask, 1 << (r % 10));
    }
    __syncthreads();
    const int mask = hit_mask;
    parts4 += tk * 10 * n4;
    hits4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;
    const float w0 = __ldg(weights + 0);
    float4 p0 = parts4[c4];
    if (mask & 1) { const float4 hh = hits4[c4]; p0.x += hh.x; p0.y += hh.y; p0.z += hh.z; p0.w += hh.w; }
    float4 sum = make_float4(p0.x * w0, p0.y * w0, p0.z * w0, p0.w * w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        const float w = __ldg(weights + expert);
        float4 p = parts4[int64_t(expert) * n4 + c4];
        if ((mask >> expert) & 1) { const float4 hh = hits4[int64_t(expert) * n4 + c4]; p.x += hh.x; p.y += hh.y; p.z += hh.z; p.w += hh.w; }
        sum.x = fmaf(p.x, w, sum.x);
        sum.y = fmaf(p.y, w, sum.y);
        sum.z = fmaf(p.z, w, sum.z);
        sum.w = fmaf(p.w, w, sum.w);
    }
    const float4 sh = shared4[c4];
    sum.x += sh.x;
    sum.y += sh.y;
    sum.z += sh.z;
    sum.w += sh.w;
    output4[c4] = sum;
}
} // namespace legacy
// GATED (S26 STRATA_LFUSE): `shared` is the shared expert's unscaled output and sg its raw gate logit per token;
// the row is scaled here as shared_expert_multi did (native_scalar_sigmoid_multi_kernel's expression, then one
// rounded multiply = scale_rows_kernel's `out *= g`), then added as before
template<bool ZADD = false, bool GATED = false>
__global__ void combine(const float* __restrict__ parts, const float* __restrict__ weights,
                        const float* __restrict__ shared, float* __restrict__ output,
                        int64_t n_embd, int k, const float* __restrict__ sg = nullptr) {
    // blockIdx.y = the token of a multi-token launch (0 for the single one)
    const int64_t tk = blockIdx.y;
    parts += tk * k * n_embd; weights += tk * k; if (shared) shared += tk * n_embd; output += tk * n_embd;
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    // ZADD: a part is 0.0f + hit (the zeroed row plus moe_hit_add of the verify window), rounded as that add
    float p0 = parts[col];
    if (ZADD) p0 = ordered_moe::add_rn(0.0f, p0);
    ordered_moe::Fold sum(p0, weights[0]);
    for (int expert = 1; expert < k; ++expert) {
        float p = parts[int64_t(expert) * n_embd + col];
        if (ZADD) p = ordered_moe::add_rn(0.0f, p);
        sum.append(p, weights[expert]);
    }
    if constexpr (GATED) {
        // Keep the existing sigmoid, but round its scale_rows-equivalent
        // product before the final shared add, including under NVCC fast math.
        const float g = __fdividef(1.0f, 1.0f + __expf(-sg[tk]));
        output[col] = sum.with_shared(ordered_moe::mul_rn(shared[col], g));
    } else {
        output[col] = shared ? sum.with_shared(shared[col]) : sum.value();
    }
}
__global__ void combine_k10_vec4(const float4* __restrict__ parts4, const float* __restrict__ weights,
                                 const float4* __restrict__ shared4, float4* __restrict__ output4,
                                 int64_t n4) {
    const int64_t tk = blockIdx.y;
    parts4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;
    const float w0 = __ldg(weights + 0);
    const float4 p0 = parts4[c4];
    ordered_moe::Fold sx(p0.x, w0), sy(p0.y, w0), sz(p0.z, w0), sw(p0.w, w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        const float w = __ldg(weights + expert);
        const float4 p = parts4[int64_t(expert) * n4 + c4];
        sx.append(p.x, w); sy.append(p.y, w); sz.append(p.z, w); sw.append(p.w, w);
    }
    const float4 sh = shared4[c4];
    output4[c4] = make_float4(sx.with_shared(sh.x), sy.with_shared(sh.y),
                              sz.with_shared(sh.z), sw.with_shared(sh.w));
}
__global__ void combine_k10_vec4_hits(const float4* __restrict__ parts4, const float4* __restrict__ hits4,
                                      const int32_t* __restrict__ dst, const int32_t* __restrict__ count,
                                      const float* __restrict__ weights, const float4* __restrict__ shared4,
                                      float4* __restrict__ output4, int64_t n4) {
    const int64_t tk = blockIdx.y;
    __shared__ int hit_mask;
    if (threadIdx.x == 0) hit_mask = 0;
    __syncthreads();
    const int nh = *count;
    for (int h = threadIdx.x; h < nh; h += blockDim.x) {
        const int r = dst[h];
        if (r / 10 == (int) tk) atomicOr(&hit_mask, 1 << (r % 10));
    }
    __syncthreads();
    const int mask = hit_mask;
    parts4 += tk * 10 * n4;
    hits4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;
    const float w0 = __ldg(weights + 0);
    float4 p0 = parts4[c4];
    if (mask & 1) {
        const float4 hh = hits4[c4];
        p0.x = ordered_moe::add_rn(p0.x, hh.x); p0.y = ordered_moe::add_rn(p0.y, hh.y);
        p0.z = ordered_moe::add_rn(p0.z, hh.z); p0.w = ordered_moe::add_rn(p0.w, hh.w);
    }
    ordered_moe::Fold sx(p0.x, w0), sy(p0.y, w0), sz(p0.z, w0), sw(p0.w, w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        const float w = __ldg(weights + expert);
        float4 p = parts4[int64_t(expert) * n4 + c4];
        if ((mask >> expert) & 1) {
            const float4 hh = hits4[int64_t(expert) * n4 + c4];
            p.x = ordered_moe::add_rn(p.x, hh.x); p.y = ordered_moe::add_rn(p.y, hh.y);
            p.z = ordered_moe::add_rn(p.z, hh.z); p.w = ordered_moe::add_rn(p.w, hh.w);
        }
        sx.append(p.x, w); sy.append(p.y, w); sz.append(p.z, w); sw.append(p.w, w);
    }
    const float4 sh = shared4[c4];
    output4[c4] = make_float4(sx.with_shared(sh.x), sy.with_shared(sh.y),
                              sz.with_shared(sh.z), sw.with_shared(sh.w));
}
// Host-only selection: graph capture records the chosen kernel, with no
// per-output device branch or additional launch in either numerical profile.
template<bool ZADD = false, bool GATED = false>
void launch_combine(dim3 grid, const float* parts, const float* weights, const float* shared,
                    float* output, int64_t n_embd, int k, void* stream, const float* sg = nullptr) {
    if (batch_deterministic_reduction_enabled())
        combine<ZADD, GATED><<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(parts, weights, shared, output, n_embd, k, sg);
    else
        legacy::combine<ZADD, GATED><<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(parts, weights, shared, output, n_embd, k, sg);
}
void launch_k10(dim3 grid, const float4* parts, const float* weights, const float4* shared,
                float4* output, int64_t n4, void* stream) {
    if (batch_deterministic_reduction_enabled())
        combine_k10_vec4<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(parts, weights, shared, output, n4);
    else
        legacy::combine_k10_vec4<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(parts, weights, shared, output, n4);
}
void launch_k10_hits(dim3 grid, const float4* parts, const float4* hits, const int32_t* dst, const int32_t* count,
                     const float* weights, const float4* shared, float4* output, int64_t n4, void* stream) {
    if (batch_deterministic_reduction_enabled())
        combine_k10_vec4_hits<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(parts, hits, dst, count, weights, shared, output, n4);
    else
        legacy::combine_k10_vec4_hits<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(parts, hits, dst, count, weights, shared, output, n4);
}
bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes)
            || (shared && !valid_span(shared, row_bytes))
            || overlap(output, row_bytes, parts, part_bytes)
            || overlap(output, row_bytes, weights, weight_bytes)
            || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    if (k == 10 && shared != nullptr && (n_embd & 3) == 0 &&
        (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0) {
        const int64_t n4 = n_embd >> 2;
        launch_k10(dim3(unsigned((n4 + 127) / 128)),
            reinterpret_cast<const float4*>(parts), weights, reinterpret_cast<const float4*>(shared),
            reinterpret_cast<float4*>(output), n4, stream);
    } else {
        launch_combine(dim3(unsigned((n_embd + 255) / 256)),
            parts, weights, shared, output, n_embd, int(k), stream);
    }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
bool native_moe_combine_hits_supported(const float* parts, const float* shared, const float* output, int64_t n_embd, int64_t k, int n_tok) {
    return k == 10 && shared != nullptr && n_tok >= 1 && n_tok <= 8 && n_embd > 0 && (n_embd & 3) == 0 &&
           (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0;
}
void native_moe_combine_multi_hits_added(const float* parts, const float* hits, const int32_t* dst, const int32_t* count,
                                         const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k,
                                         int n_tok, void* stream) {
    if (!stream || !native_moe_combine_hits_supported(parts, shared, output, n_embd, k, n_tok) || ((uintptr_t) hits & 15u) != 0)
        throw std::invalid_argument("native MoE combine (hits added) requires k = 10, aligned spans, 1..8 tokens");
    const int64_t n4 = n_embd >> 2;
    launch_k10_hits(dim3(unsigned((n4 + 127) / 128), (unsigned) n_tok),
        reinterpret_cast<const float4*>(parts), reinterpret_cast<const float4*>(hits), dst, count, weights,
        reinterpret_cast<const float4*>(shared), reinterpret_cast<float4*>(output), n4, stream);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    if (k == 10 && shared != nullptr && (n_embd & 3) == 0 &&
        (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0) {
        const int64_t n4 = n_embd >> 2;
        launch_k10(dim3(unsigned((n4 + 127) / 128), (unsigned) n_tok),
            reinterpret_cast<const float4*>(parts), weights, reinterpret_cast<const float4*>(shared),
            reinterpret_cast<float4*>(output), n4, stream);
    } else {
        launch_combine(dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok),
            parts, weights, shared, output, n_embd, int(k), stream);
    }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_multi_hits_gated(const float* hits, const float* weights, const float* shared,
                                         const float* shared_gate, float* output, int64_t n_embd, int64_t k, int n_tok,
                                         void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1 || !shared || !shared_gate)
        throw std::invalid_argument("native MoE combine (hits, gated) requires a stream, width, 1..15 experts, tokens");
    launch_combine<true, true>(dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok),
        hits, weights, shared, output, n_embd, int(k), stream, shared_gate);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_multi_hits(const float* hits, const float* weights, const float* shared, float* output,
                                   int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (hits) requires a stream, width, 1..15 experts, tokens");
    launch_combine<true>(dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok),
        hits, weights, shared, output, n_embd, int(k), stream);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
