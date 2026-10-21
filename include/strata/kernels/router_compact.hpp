#pragma once
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_ROUTER_HD __host__ __device__
#else
#define STRATA_ROUTER_HD
#endif

namespace strata::kernels::router_compact {
inline constexpr int experts = 512, lanes = 32, values_per_lane = 16, topk = 10;
inline constexpr int groups = 16, capacity = 128;
static_assert(groups >= topk);

// These are integer prefixes of the ORIGINAL normalized FP32 probabilities,
// not rounded probabilities and not a replacement router GEMM. Ordering is
// monotonic only for nonnegative finite FP32. Subnormals also use the legacy
// path: an FTZ-enabled floating comparison may treat them as zero, unlike the
// integer order. No subnormal float arithmetic is used for control anywhere.
STRATA_ROUTER_HD inline constexpr bool valid_probability_bits(uint32_t bits) {
    return bits == 0 || (bits >= 0x00800000u && bits < 0x7f800000u);
}
STRATA_ROUTER_HD inline constexpr uint16_t bucket(uint32_t bits) {
    return static_cast<uint16_t>(bits >> 16);
}
STRATA_ROUTER_HD inline constexpr bool keep(uint32_t bits, uint16_t threshold) {
    return bucket(bits) >= threshold;
}
STRATA_ROUTER_HD inline constexpr bool usable(int count, bool valid) {
    return valid && count >= groups && count <= capacity;
}

// Strict process-level switch; set before graph capture. The default is off.
inline bool parse_enabled(const char* text, bool& enabled) {
    enabled = false;
    if (!text) return true;
    if ((text[0] != '0' && text[0] != '1') || text[1] != '\0') return false;
    enabled = text[0] == '1';
    return true;
}

// Proof: pair lanes j and j+16, then take the MINIMUM of the 16 group maxima.
// Each disjoint group contributes a value at or above it, so at least 16
// candidates survive (>= top 10). A lower bucket cannot contain a true winner.
// Keep the entire boundary bucket: truncating it would break ties/exactness.
} // namespace strata::kernels::router_compact

#undef STRATA_ROUTER_HD
