// Public CPU contract for the compact router's integer bound and selection.
// No CUDA/model assets. GPU normalization and capture parity have a separate test.
#include "strata/kernels/router_compact.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace rc = strata::kernels::router_compact;
namespace {
size_t checks = 0, rows = 0, compact_rows = 0, fallback_rows = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr,"FAIL %s at row=%zu check=%zu\n",what,rows,checks); std::exit(1); }
}
using Bits = std::array<uint32_t,rc::experts>;
using Picks = std::array<int,rc::topk>;
float value(uint32_t bits) { return std::bit_cast<float>(bits); }
uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }
struct Plan { uint16_t threshold = 0; int count = 0; bool compact = false; std::vector<int> ids; };
Plan plan(const Bits& p) {
    Plan result;
    result.threshold = UINT16_MAX;
    bool valid = true;
    std::array<uint16_t,rc::lanes> lane_max{};
    for (int lane = 0; lane < rc::lanes; ++lane) {
        uint16_t maximum = 0;
        for (int i = 0; i < rc::values_per_lane; ++i) {
            const auto b = p[lane+i*rc::lanes];
            valid = valid && rc::valid_probability_bits(b);
            maximum = std::max(maximum,rc::bucket(b));
        }
        lane_max[lane] = maximum;
    }
    for (int group = 0; group < rc::groups; ++group)
        result.threshold = std::min(result.threshold,std::max(lane_max[group],lane_max[group+rc::groups]));
    if (!valid) { result.count = rc::experts; return result; }
    // Same lane-major prefix-compaction order as the device kernel.
    for (int lane = 0; lane < rc::lanes; ++lane)
        for (int i = 0; i < rc::values_per_lane; ++i) {
            const int expert = lane+i*rc::lanes;
            if (rc::keep(p[expert],result.threshold)) result.ids.push_back(expert);
        }
    result.count = static_cast<int>(result.ids.size());
    check(result.count >= rc::groups,"every disjoint group contributes a survivor");
    result.compact = rc::usable(result.count,valid);
    return result;
}
Picks oracle(const Bits& p) {
    std::array<int,rc::experts> ids{};
    std::iota(ids.begin(),ids.end(),0);
    std::partial_sort(ids.begin(),ids.begin()+rc::topk,ids.end(),[&](int a,int b) {
        return p[a] != p[b] ? p[a] > p[b] : a < b;
    });
    Picks out{}; std::copy_n(ids.begin(),rc::topk,out.begin()); return out;
}
// Model the device's four-candidate-per-lane warp reduction independently of
// the full sort oracle. Ties always compare ORIGINAL expert IDs.
Picks compact_select(const Bits& p, const Plan& selected) {
    auto ids = selected.ids;
    Picks out{};
    for (int rank = 0; rank < rc::topk; ++rank) {
        std::array<int,rc::lanes> local{}; local.fill(-1);
        auto better = [&](int a,int b) {
            if (a < 0) return false;
            if (b < 0) return true;
            return p[a] != p[b] ? p[a] > p[b] : a < b;
        };
        for (int lane = 0; lane < rc::lanes; ++lane)
            for (int i = 0; i < rc::capacity/rc::lanes; ++i) {
                const size_t at = lane+i*rc::lanes;
                if (at < ids.size() && better(ids[at],local[lane])) local[lane] = ids[at];
            }
        for (int mask = 16; mask; mask >>= 1) {
            const auto before = local;
            for (int lane = 0; lane < rc::lanes; ++lane)
                if (better(before[lane^mask],local[lane])) local[lane] = before[lane^mask];
        }
        check(local[0] >= 0,"compact select has enough real winners");
        for (int lane = 1; lane < rc::lanes; ++lane) check(local[lane] == local[0],"warp agrees on winner");
        out[rank] = local[0];
        *std::find(ids.begin(),ids.end(),out[rank]) = -1;
    }
    return out;
}
float add(float a,float b) { volatile float out = a+b; return out; }
std::array<uint32_t,rc::topk> weights(const Bits& p,const Picks& picks,bool original_lanes) {
    std::array<float,rc::lanes> sum{};
    for (int rank = 0; rank < rc::topk; ++rank) {
        const int lane = original_lanes ? picks[rank]%rc::lanes : rank;
        sum[lane] = add(sum[lane],value(p[picks[rank]]));
    }
    for (int mask = 16; mask; mask >>= 1) {
        const auto before = sum;
        for (int lane = 0; lane < rc::lanes; ++lane) sum[lane] = add(before[lane],before[lane^mask]);
    }
    const float reciprocal = 1.0f/std::max(sum[0],6.103515625e-5f);
    std::array<uint32_t,rc::topk> out{};
    for (int rank = 0; rank < rc::topk; ++rank) out[rank] = bits(value(p[picks[rank]])*reciprocal);
    return out;
}
void verify(const Bits& p) {
    ++rows;
    const auto selected = plan(p);
    check(std::all_of(p.begin(),p.end(),rc::valid_probability_bits),"fixture finite zero/normal probabilities");
    const auto expected = oracle(p);
    if (selected.compact) {
        ++compact_rows;
        check(selected.count <= rc::capacity,"compact scratch capacity");
        for (int expert : expected)
            check(std::find(selected.ids.begin(),selected.ids.end(),expert) != selected.ids.end(),"no false-negative winner");
        const auto actual = compact_select(p,selected);
        check(actual == expected,"exact ranked IDs, including boundary ties");
        check(weights(p,actual,true) == weights(p,expected,true),"original-lane normalized FP32 weights");
    } else {
        ++fallback_rows;
        check(selected.count > rc::capacity,"dense rows take the full legacy selection");
    }
}
} // namespace

int main() {
    bool enabled = true;
    check(rc::parse_enabled(nullptr,enabled) && !enabled,"unset stays off");
    check(rc::parse_enabled("0",enabled) && !enabled,"explicit off");
    check(rc::parse_enabled("1",enabled) && enabled,"explicit on");
    for (const char* text : {"","true","false","01","10","-1"," 1","1 ","2"})
        check(!rc::parse_enabled(text,enabled) && !enabled,"strict flag rejects malformed text");
    for (int n = -1; n <= 513; ++n) {
        check(rc::usable(n,true) == (n >= 16 && n <= 128),"capacity predicate");
        check(!rc::usable(n,false),"invalid row cannot compact");
    }
    Bits p{};
    for (uint32_t invalid : {1u,0x007fffffu,0x80000000u,0xbf800000u,0x7f800000u,0xff800000u,0x7fc00000u})
        for (int at : {0,31,32,255,511}) {
            p.fill(bits(0.125f)); p[at] = invalid;
            const auto before = p; const auto result = plan(p);
            check(!result.compact && result.count == 512,"subnormal/negative/NaN/inf exact fallback");
            check(p == before,"planning never mutates its input");
        }
    check(rc::valid_probability_bits(0) && rc::valid_probability_bits(0x00800000u),"zero and smallest normal accepted");
    p.fill(bits(1.0f/512)); verify(p);
    p.fill(0); verify(p);
    for (int n : {16,32,127,128,129,256,512}) {
        p.fill(0);
        for (int i = 0; i < n; ++i) p[i] = bits(0.125f);
        const auto result = plan(p);
        check(result.count == n && result.compact == (n <= 128),"128/129 exact boundary"); verify(p);
    }
    // Negative control: a BF16-prefix-only TopK would choose IDs0..9 and lose
    // genuine winners. The conservative filter retains the entire tied bucket.
    p.fill(0);
    for (int lane = 0; lane < 32; ++lane) p[lane] = 0x3e000000u+uint32_t(lane);
    check(oracle(p)[0] == 31 && rc::bucket(p[0]) == rc::bucket(p[31]),"naive low-precision rank red control");
    verify(p);
    // Negative control for the easily missed normalization reduction contract.
    p.fill(0);
    p[0] = bits(0.5f);
    for (int i = 1; i < 10; ++i) p[i*32] = bits(0x1p-26f);
    for (int lane = 1; lane < 32; ++lane) p[lane] = bits(0x1p-29f);
    const auto exact = oracle(p);
    check(weights(p,exact,true) != weights(p,exact,false),"rank-lane summation must fail bitwise control");
    verify(p);
    std::mt19937 random(5090);
    for (int row = 0; row < 6000; ++row) {
        for (int i = 0; i < rc::experts; ++i) {
            const int exponent = 87+int(random()%40); // all normal FP32, broad/sparse/dense buckets
            p[i] = (uint32_t(exponent)<<23) | (random()&0x007fffffu);
            if (row%3 == 0) p[i] = 0x3d000000u+(random()&0xffffu);
            if (row%7 == 0 && i%5 == 0) p[i] = 0;
        }
        verify(p);
    }
    check(compact_rows > 100 && fallback_rows > 100,"both branches exercised");
    std::printf("PASS router compact CPU: %zu checks, %zu rows (%zu compact / %zu full fallback); "
                "exact Top10/ties, original-lane weights, integer/FTZ guards and red controls; NOT GPU\n",
                checks,rows,compact_rows,fallback_rows);
}
