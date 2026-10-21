// Public fixed-leaf contract. This does not execute or equate CPU/GPU experts.
#include "strata/kernels/ordered_moe_fold.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace om = strata::kernels::ordered_moe;
namespace {
size_t checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr,"FAIL ordered MoE fold: %s (check=%zu)\n",label,checks); std::exit(1); }
}
uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }
float from_bits(uint32_t x) { return std::bit_cast<float>(x); }
float ref_mul(float a, float b) { volatile float x = a*b; return x; }
float ref_add(float a, float b) { volatile float x = a+b; return x; }
float reference(const float* parts, const float* weights, int k, bool shared, float sh) {
    float result = ref_mul(parts[0],weights[0]);
    for (int i = 1; i < k; ++i) result = std::fma(parts[i],weights[i],result);
    return shared ? ref_add(result,sh) : result;
}
float folded(const float* parts, const float* weights, int k, bool shared, float sh) {
    om::Fold result(parts[0],weights[0]);
    for (int i = 1; i < k; ++i) result.append(parts[i],weights[i]);
    return shared ? result.with_shared(sh) : result.value();
}
void same(float a, float b, const char* label) { check(bits(a)==bits(b),label); }
void boundaries() {
    const float tiny = from_bits(1);
    same(om::add_rn(tiny,tiny),from_bits(2),"host environment must preserve subnormals");
    same(om::mul_rn(from_bits(3),0.5f),from_bits(2),"RN underflow halfway rounds even");
    const float a = from_bits(0x3f800001), b = from_bits(0x3f7ffffe);
    om::Fold fused(-1.0f,1.0f); fused.append(a,b);
    same(fused.value(),-std::ldexp(1.0f,-46),"expert append must be a fused multiply-add");
    check(bits(fused.value())!=bits(ref_add(ref_mul(a,b),-1.0f)),"positive control detects unfused append");
    om::Fold first(a,b);
    same(first.with_shared(-1.0f),0.0f,"first product and final shared add round separately");
    check(bits(first.with_shared(-1.0f))!=bits(std::fma(a,b,-1.0f)),"positive control detects first/shared fusion");
    om::Fold gated(tiny,1.0f);
    same(gated.with_shared(om::mul_rn(from_bits(3),0.5f)),from_bits(3),"shared scaling rounds before shared add");
    check(bits(gated.with_shared(om::mul_rn(from_bits(3),0.5f)))!=bits(std::fma(from_bits(3),0.5f,tiny)),
          "positive control detects gated shared fusion");
    om::Fold order(1e20f,1.0f); order.append(-1e20f,1.0f); order.append(3.0f,1.0f);
    om::Fold other(1e20f,1.0f); other.append(3.0f,1.0f); other.append(-1e20f,1.0f);
    same(order.value(),3.0f,"router-rank ordered cancellation");
    check(bits(order.value())!=bits(other.value()),"positive control detects fold reassociation");
    for (float zero : {0.0f,-0.0f}) for (float w : {1.0f,-1.0f}) {
        om::Fold one(zero,w);
        same(one.value(),ref_mul(zero,w),"signed zero first product");
        for (float shared : {0.0f,-0.0f}) same(one.with_shared(shared),ref_add(ref_mul(zero,w),shared),"signed shared zero");
    }
    // Deliberately different leaves are NOT covered by batch invariance. A
    // fixed sum tree cannot repair different expert quantizers/nonlinearities.
    om::Fold cpu(1.0f,1.0f), gpu(std::nextafter(1.0f,2.0f),1.0f);
    check(bits(cpu.value())!=bits(gpu.value()),"changed expert leaf must remain observable");
    const float competitor = std::nextafter(1.0f,2.0f);
    const int cpu_argmax = cpu.value() >= competitor ? 0 : 1;
    const int gpu_argmax = gpu.value() >= competitor ? 0 : 1;
    check(cpu_argmax!=gpu_argmax,"one-ULP producer delta can change greedy token despite fixed fold");
}
void random_reference() {
    std::mt19937 rng(0x50902026);
    std::array<float,15> parts{}, weights{};
    for (int k = 1; k <= 15; ++k) for (int trial = 0; trial < 1024; ++trial) {
        for (int i = 0; i < k; ++i) {
            parts[i] = std::ldexp(float(int(rng()%2001)-1000),int(rng()%81)-40);
            weights[i] = float(int(rng()%1025)-512)/512.0f;
        }
        const float sh = float(int(rng()%2001)-1000)/8.0f;
        for (bool shared : {false,true})
            same(folded(parts.data(),weights.data(),k,shared,sh),reference(parts.data(),weights.data(),k,shared,sh),
                 "scalar fold matches independent rank-ordered RN reference");
    }
    const std::array<uint32_t,12> adversarial = {1,2,3,0x007fffff,0x00800000,0x3f000000,
        0x3f7fffff,0x3f800001,0x80000001,0x807fffff,0xbf800001,0xc0000000};
    for (uint32_t a : adversarial) for (uint32_t b : adversarial) for (uint32_t c : adversarial) {
        same(om::add_rn(from_bits(a),from_bits(b)),ref_add(from_bits(a),from_bits(b)),"adversarial add");
        same(om::mul_rn(from_bits(a),from_bits(b)),ref_mul(from_bits(a),from_bits(b)),"adversarial multiply");
        same(om::fma_rn(from_bits(a),from_bits(b),from_bits(c)),std::fma(from_bits(a),from_bits(b),from_bits(c)),
             "adversarial FMA");
    }
}
void mixed_arrival_and_lanes() {
    constexpr int k = 10, owners = 4, rows = 12, columns = 8;
    std::array<float,owners*k*columns> leaves{};
    std::array<float,owners*k> weights{};
    std::array<float,owners*columns> shared{};
    for (size_t i = 0; i < leaves.size(); ++i) leaves[i] = float(int((i*31)%113)-56)/16.0f;
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = float((i*13)%37+1)/64.0f;
    for (size_t i = 0; i < shared.size(); ++i) shared[i] = float(int(i%19)-9)/32.0f;
    std::array<int,rows> row_owner{3,3,3,1,1,1,0,0,0,2,2,2};
    std::array<int,k> arrival{}; std::iota(arrival.begin(),arrival.end(),0);
    std::mt19937 rng(0xc40104);
    // All 1024 CPU/GPU ownership subsets; arrival order varies independently.
    // Fixed leaves are written into their logical rank, never completion order.
    for (int mask = 0; mask < (1<<k); ++mask) {
        std::shuffle(arrival.begin(),arrival.end(),rng);
        for (int active : {1,4,1}) for (int row = 0; row < (active==1 ? 1 : rows); ++row) {
            const int owner = active==1 ? 3 : row_owner[row];
            for (int col = 0; col < columns; ++col) {
                std::array<float,k> cpu{}, hit{}, logical{}, mixed{}, ws{};
                for (int rank : arrival) {
                    logical[rank] = leaves[(owner*k+rank)*columns+col];
                    ws[rank] = weights[owner*k+rank];
                    if ((mask>>rank)&1) hit[rank] = logical[rank]; else cpu[rank] = logical[rank];
                }
                for (int rank = 0; rank < k; ++rank)
                    mixed[rank] = ((mask>>rank)&1) ? om::add_rn(cpu[rank],hit[rank]) : cpu[rank];
                const float sh = shared[owner*columns+col];
                same(folded(mixed.data(),ws.data(),k,true,sh),reference(logical.data(),ws.data(),k,true,sh),
                     "same leaves invariant to ownership/arrival and 1->4->1 row grouping");
            }
        }
    }
}
} // namespace
int main() {
    check(std::fesetround(FE_TONEAREST)==0 && std::fegetround()==FE_TONEAREST,"round-to-nearest host witness");
    boundaries(); random_reference(); mixed_arrival_and_lanes();
    std::printf("PASS ordered_moe_fold_cpu: %zu checks; rank order, RN/FMA/FTZ boundaries, 1024 fixed-leaf ownership masks, "
                "1->4->1 grouping; changed-leaf counterexample retained. NOT model/GPU expert parity.\n",checks);
}
