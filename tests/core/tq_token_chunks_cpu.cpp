// Merged expert jobs may contain 16 tokens although TQ ISA scratch holds eight.
// Exercise the public entries, not a duplicate chunk helper. No TQ ggml traits or model/GPU needed.
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace {
size_t checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s (check %zu)\n", what, checks); std::exit(1); }
}
constexpr int rows = 9, stride = rows + 2;
constexpr float poison = std::bit_cast<float>(uint32_t{0x7fc12345});

void run(cpu::TqIsa isa, int type, int n, bool hadamard) {
    constexpr int tokens = 16;
    const size_t row_bytes = size_t(n / cpu::kTqBlock) * cpu::tq_block_bytes(type);
    const size_t up_off = rows * row_bytes + 32; // nonzero gap must not change when token pointers move
    std::vector<uint8_t> weights(up_off + rows * row_bytes + 32, 0xc7);
    std::mt19937 rng(1000 + type + n);
    for (size_t base : {size_t{0}, up_off})
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < n / cpu::kTqBlock; ++b) {
                auto* p = weights.data() + base + r * row_bytes + b * cpu::tq_block_bytes(type);
                const uint16_t scale = uint16_t(0x1800 + 7 * r + b);
                std::memcpy(p, &scale, 2);
                for (size_t j = 2; j < cpu::tq_block_bytes(type); ++j) p[j] = uint8_t(rng());
            }
    const auto original_weights = weights;
    const size_t qbytes = ggml_row_size(GGML_TYPE_Q8_0, n);
    std::vector<uint8_t> acts(qbytes * tokens);
    std::vector<const void*> ap(tokens);
    std::vector<float> x(n), signs(n);
    for (int j = 0; j < n; ++j) signs[j] = (j % 3) ? 1.f : -1.f;
    for (int t = 0; t < tokens; ++t) {
        for (int j = 0; j < n; ++j) x[j] = std::sin(float(j + 13 * t) * .071f) * (.1f + .01f * t);
        if (hadamard) cpu::hadamard_rotate(x.data(), n, 128, signs.data());
        auto* q = acts.data() + t * qbytes;
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x.data(), q, n);
        ap[t] = hadamard && t % 5 == 4 ? ap[t - 1] : q; // repeated input pointers are legal
    }
    const auto original_acts = acts;
    // Wide -> narrow -> wide also exercises reused thread-local scratch, partial tails and nonzero r0.
    for (const int nt : {16, 1, 9, 2, 12, 3, 15, 4, 10, 5, 14, 6, 11, 7, 13, 8})
        for (const int r0 : {0, 2})
            for (const bool gu : {false, true}) {
                const int r1 = r0 ? 7 : rows;
                std::vector<float> ref(tokens * stride, poison), got(ref), split(ref);
                std::vector<float*> rp(tokens), gp(tokens), sp(tokens);
                for (int t = 0; t < tokens; ++t) {
                    const int physical = (t * 5) % tokens; // non-contiguous output pointer table
                    rp[t] = ref.data() + physical * stride + 1;
                    gp[t] = got.data() + physical * stride + 1;
                    sp[t] = split.data() + physical * stride + 1;
                }
                auto call = [&](int first, int count, float* const* out) {
                    if (gu) cpu::tq_gu_rows(isa, type, weights.data(), row_bytes, up_off, n,
                                            ap.data() + first, count, out + first, r0, r1);
                    else cpu::tq_rows(isa, type, weights.data(), row_bytes, n,
                                      ap.data() + first, count, out + first, r0, r1);
                };
                for (int t = 0; t < nt; ++t) call(t, 1, rp.data());
                // Existing safe <=8 calls provide a second independent public-entry baseline.
                const int first_count = nt < 8 ? nt : 8;
                call(0, first_count, sp.data());
                if (nt > 8) call(8, nt - 8, sp.data());
                call(0, nt, gp.data());
                check(std::memcmp(ref.data(), got.data(), got.size() * sizeof(float)) == 0, "group vs scalar bits/guards");
                check(std::memcmp(split.data(), got.data(), got.size() * sizeof(float)) == 0, "group vs <=8 chunks bits/guards");
                for (int t = 0; t < nt; ++t)
                    for (int r = r0; r < r1; ++r) check(std::isfinite(gp[t][r]), "finite output");
                check(weights == original_weights && acts == original_acts, "inputs immutable");
            }
}
} // namespace

int main() {
    ggml_cpu_init();
    int isas = 0;
    for (auto isa : {cpu::TqIsa::scalar, cpu::TqIsa::avx2, cpu::TqIsa::avx512}) {
        if (!cpu::tq_isa_ok(isa)) { std::printf("ISA %s unavailable (not executed)\n", cpu::tq_isa_name(isa)); continue; }
        ++isas;
        for (int type : {144, 145, 146})
            for (int n : {128, 640, 2560})
                for (bool hadamard : {false, true}) run(isa, type, n, hadamard);
        std::printf("ISA %s PASS\n", cpu::tq_isa_name(isa));
    }
    std::printf("%zu checks PASS; %d ISA(s); tokens 1..16, GU/down, Hadamard inputs, nonzero row ranges and guards\n", checks, isas);
    return 0;
}
