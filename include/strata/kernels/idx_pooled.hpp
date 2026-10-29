// include/strata/kernels/idx_pooled.hpp - C8-6 STRATA_IDX_POOLED_BF16=1: the QSA indexer's pooled keys in BF16.
//
// A pooled row is idx_dim = 128 keys: 512 B in FP32 (the default), 256 B in BF16.  At 262144 cells one session's 12
// QSA layers hold 12 x 65,538 rows - 384 MiB in FP32, 192 MiB in BF16 - so C8's eight slot sessions give 1,536 MiB
// back to the expert cache.  Lossy: the selection may change; it is opt-in and gated on quality.
//
// The storage is a process-wide mode fixed before the first session is sized; it never changes while one exists.
// `QsaState::idx_pooled` keeps its `float*` type as the STORAGE HANDLE (the struct and DeviceQsaSlot are shared with
// the SYCL and Metal trees): host code never indexes it - byte counts come from idx_pooled_bytes(), and a kernel
// reads or writes it only through pooled_load4/pooled_load/pooled_store<BF> with an element index, BF taken from
// qsa_idx_pooled_bf16() at launch (so at capture).  Only what the indexer stores is rounded, once, to nearest even:
// the completed rows and the spare copy of `dead`.  The raw tail and `dead` itself stay FP32, and so do the pooling
// arithmetic, the queries and the scores: a BF16 key widens exactly and enters the FP32 code unchanged, so a BF16
// call scores bit for bit what the FP32 call scores on the widened keys.  Off, every kernel is the FP32 one it was.
#pragma once

#include "strata/kernels/bf16_bits.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#if defined(__HIPCC__) && defined(STRATA_HIP_GFX906)   // as mrope.hpp: the gfx906 compat build parses this first
#include <hip/hip_runtime.h>
#endif

namespace strata::kernels {

inline std::atomic<bool>& idx_pooled_bf16_mode() {
    static std::atomic<bool> on{false};
    return on;
}
/// Before any session is sized (generate.cpp, STRATA_IDX_POOLED_BF16).
inline void qsa_set_idx_pooled_bf16(bool on) { idx_pooled_bf16_mode().store(on, std::memory_order_relaxed); }
inline bool qsa_idx_pooled_bf16() { return idx_pooled_bf16_mode().load(std::memory_order_relaxed); }
/// Bytes of `rows` pooled rows of `dim` keys in the current mode.
inline uint64_t idx_pooled_bytes(uint64_t rows, uint64_t dim) { return rows * dim * (qsa_idx_pooled_bf16() ? 2 : 4); }
/// The K/V snapshot format flag of a BF16 pooled store (beside rotated INT8's +16): a snapshot, checkpoint or
/// session file of one storage never restores into the other.
constexpr int kIdxPooledBf16Format = 32;

/// Four BF16 keys read as two little-endian 32-bit words, widened: key 2j is the low half of word j.
STRATA_BF16_HD inline void pooled_widen4(uint32_t w0, uint32_t w1, float k[4]) {
    k[0] = f32_from_bf16((uint16_t) w0);
    k[1] = f32_from_bf16((uint16_t) (w0 >> 16));
    k[2] = f32_from_bf16((uint16_t) w1);
    k[3] = f32_from_bf16((uint16_t) (w1 >> 16));
}

#if defined(__CUDACC__) || defined(__HIPCC__)
/// Keys i..i+3 of the store `h` (i a multiple of 4): one 16-byte load in FP32, one 8-byte load in BF16.
template <bool BF>
__device__ __forceinline__ float4 pooled_load4(const float* h, std::size_t i) {
    if constexpr (BF) {
        const uint2 u = *reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(h) + i);
        float k[4];
        pooled_widen4(u.x, u.y, k);
        return make_float4(k[0], k[1], k[2], k[3]);
    } else {
        return *reinterpret_cast<const float4*>(h + i);
    }
}
template <bool BF>
__device__ __forceinline__ float pooled_load(const float* h, std::size_t i) {
    if constexpr (BF) return f32_from_bf16(reinterpret_cast<const uint16_t*>(h)[i]);
    else return h[i];
}
template <bool BF>
__device__ __forceinline__ void pooled_store(float* h, std::size_t i, float y) {
    if constexpr (BF) reinterpret_cast<uint16_t*>(h)[i] = bf16_from_f32(y);   // round to nearest even, once
    else h[i] = y;
}
#endif

}  // namespace strata::kernels
