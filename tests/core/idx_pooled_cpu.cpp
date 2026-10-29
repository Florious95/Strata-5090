// C8-6 STRATA_IDX_POOLED_BF16: the host side of the BF16 pooled store (idx_pooled.hpp) - the process mode and byte
// counts, the capacity it gives back at 262144 cells, the BF16 group layout the scorers widen, the rounding every
// write applies (the indexer's rows and the restore's spare row) and the snapshot format codes. No GPU.
#include "strata/kernels/idx_pooled.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <vector>

namespace k = strata::kernels;
namespace { long checks = 0; }
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL %s (check %ld)\n", what, checks); std::exit(1); }
}
static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float from_bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

int main() {
    // ---- the mode: off by default, FP32 bytes; on, half; the byte count follows the mode, nothing else
    check(!k::qsa_idx_pooled_bf16(), "default off");
    check(k::idx_pooled_bytes(65538, 128) == 65538ull * 128 * 4, "fp32 bytes");
    k::qsa_set_idx_pooled_bf16(true);
    check(k::qsa_idx_pooled_bf16() && k::idx_pooled_bytes(65538, 128) == 65538ull * 128 * 2, "bf16 bytes");
    check(k::idx_pooled_bytes(0, 128) == 0 && k::idx_pooled_bytes(2, 128) == 512, "ring rows");
    k::qsa_set_idx_pooled_bf16(false);
    check(!k::qsa_idx_pooled_bf16() && k::idx_pooled_bytes(2, 128) == 1024, "back off");

    // ---- capacity at the production context: layer.cpp's rows (max_cells / 4 + 2) x 12 QSA layers per session
    const uint64_t rows = 262144 / 4 + 2, layers = 12, expert = 1305600;
    k::qsa_set_idx_pooled_bf16(false);
    const uint64_t f32 = layers * k::idx_pooled_bytes(rows, 128);
    k::qsa_set_idx_pooled_bf16(true);
    const uint64_t b16 = layers * k::idx_pooled_bytes(rows, 128);
    k::qsa_set_idx_pooled_bf16(false);
    const double mib = 1048576.0;
    check(f32 == 402665472ull && b16 == 201332736ull, "per-session bytes");
    check(std::fabs((f32 - b16) * 4 / mib - 768.0234375) < 1e-9, "C4 saves 768.023 MiB");
    check(std::fabs((f32 - b16) * 8 / mib - 1536.046875) < 1e-9, "C8 saves 1536.047 MiB");
    check((f32 - b16) * 8 / expert == 1233, "C8: +1,233 resident experts");

    // ---- every BF16 pattern widens exactly and narrows back to itself (NaNs: quiet, same sign)
    for (uint32_t h = 0; h < 65536; ++h) {
        const float f = k::f32_from_bf16((uint16_t) h);
        check(bits(f) == h << 16, "widen is the top half");
        const bool nan = (h & 0x7fffu) > 0x7f80u;
        check(k::bf16_from_f32(f) == (nan ? (uint16_t) (h | 0x40u) : (uint16_t) h), "narrow(widen) round trip");
    }
    // ---- the store's rounding: nearest, ties to even, carries into the exponent, overflows to infinity
    check(k::bf16_from_f32(from_bits(0x3f808000u)) == 0x3f80, "tie to even (down)");
    check(k::bf16_from_f32(from_bits(0x3f818000u)) == 0x3f82, "tie to even (up)");
    check(k::bf16_from_f32(from_bits(0x3f808001u)) == 0x3f81, "above the tie");
    check(k::bf16_from_f32(from_bits(0x3fffffffu)) == 0x4000, "carry into the exponent");
    check(k::bf16_from_f32(from_bits(0x7f7fffffu)) == 0x7f80, "max float to +inf");
    check(k::bf16_from_f32(-std::numeric_limits<float>::infinity()) == 0xff80, "-inf");
    check(k::bf16_from_f32(from_bits(0x00000001u)) == 0x0000, "smallest denormal to +0");
    check(k::bf16_from_f32(-0.0f) == 0x8000, "-0");
    std::mt19937 rng(20261029);
    std::uniform_int_distribution<uint32_t> any;
    for (int i = 0; i < 1000000; ++i) {
        const uint32_t u = any(rng);
        const float f = from_bits(u);
        if (std::isnan(f)) continue;
        const uint16_t h = k::bf16_from_f32(f);
        // nearest: no other BF16 neighbour is closer (and ties take the even one)
        const double x = f, y = k::f32_from_bf16(h);
        if (std::isinf(y) && !std::isinf(x)) { check(std::fabs(x) >= 3.3895313892515355e38, "overflow only near max"); continue; }
        const double lo = k::f32_from_bf16((uint16_t) (h - 1)), hi = k::f32_from_bf16((uint16_t) (h + 1));
        const double d = std::fabs(x - y);
        if (!std::isnan(lo) && !std::isinf(lo) && (h & 0x7fffu) != 0) check(d <= std::fabs(x - lo), "nearest (below)");
        if (!std::isnan(hi) && !std::isinf(hi) && (h & 0x7fffu) != 0x7f7fu) check(d <= std::fabs(x - hi), "nearest (above)");
        // the relative error the quality gate is about: below 2^-8 for normal values
        if (std::fpclassify(f) == FP_NORMAL && !std::isinf(y)) check(d <= std::ldexp(std::fabs(x), -8), "2^-8 relative");
    }

    // ---- the group a scorer lane reads: four keys as two little-endian words (one 8-byte load) widen in key order
    std::vector<uint16_t> row(128);
    for (int r = 0; r < 64; ++r) {
        std::vector<float> want(128);
        for (int i = 0; i < 128; ++i) {
            float f;
            do f = from_bits(any(rng)); while (std::isnan(f));
            row[i] = k::bf16_from_f32(f);
            want[i] = k::f32_from_bf16(row[i]);
        }
        for (int lane = 0; lane < 32; ++lane) {
            uint32_t w[2];
            std::memcpy(w, row.data() + lane * 4, 8);   // what the device's uint2 load sees
            float got[4];
            k::pooled_widen4(w[0], w[1], got);
            check(std::memcmp(got, want.data() + lane * 4, 16) == 0, "widen4 key order");
        }
    }

    // ---- the restore's spare row (conversation_state.cpp put_spare): FP32 `dead` -> the bits the indexer itself
    // writes there (pooled_store<true> of dead[d]); the FP32 store keeps the bytes
    std::vector<float> dead(128);
    for (int i = 0; i < 128; ++i) dead[i] = std::ldexp(float(int(i % 97) - 48) / 53.0f, i % 7 - 3);
    std::vector<uint16_t> spare(128);
    for (int i = 0; i < 128; ++i) spare[i] = k::bf16_from_f32(dead[i]);
    for (int i = 0; i < 128; ++i)
        check(std::fabs(k::f32_from_bf16(spare[i]) - dead[i]) <= std::ldexp(std::fabs(dead[i]), -8), "spare row");
    check(k::idx_pooled_bytes(1, 128) == 512, "fp32 spare row is dead's 512 bytes");

    // ---- snapshot format codes (conversation_snapshot.cpp layout): base 0..3, +16 rotated F16/INT8, +32 BF16 pooled;
    // a BF16 store's code never equals an FP32 store's, so a snapshot/file of one never restores into the other
    std::set<int> fp32_codes, bf16_codes;
    for (int base = 0; base <= 3; ++base)
        for (int rot : {0, 16}) {
            if (rot && base >= 2) continue;   // Q4_0 is always rotated and K8V4 is its own code: no +16
            fp32_codes.insert(base + rot);
            bf16_codes.insert(base + rot + k::kIdxPooledBf16Format);
        }
    for (int c : bf16_codes) check(!fp32_codes.count(c), "format codes disjoint");
    check(fp32_codes.size() == 6 && bf16_codes.size() == 6, "six codes each");

    std::printf("idx_pooled_cpu: %ld checks PASS; per-session %.3f -> %.3f MiB; C8 +%llu experts\n", checks,
                f32 / mib, b16 / mib, (unsigned long long) ((f32 - b16) * 8 / expert));
}
