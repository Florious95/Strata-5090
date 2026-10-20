// Host-only Solo dispatch, logical-cap and graph-handle ownership contracts.
// Does NOT execute CUDA, attention, GDN, sampling kernels or SessionState binding.
#include "strata/core/slot_solo.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <type_traits>
using namespace strata::core;
namespace {
int checks = 0;
void require(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", label); std::exit(1); }
}
}
int main() {
    bool enabled = true;
    require(parse_slot_solo(nullptr, enabled) && !enabled, "unset is disabled");
    require(parse_slot_solo("0", enabled) && !enabled, "zero is disabled");
    require(parse_slot_solo("1", enabled) && enabled, "one is enabled");
    for (const char* invalid : {"", "true", "false", "01", "11", "1 ", "-1", "2"})
        require(!parse_slot_solo(invalid, enabled), "strict flag parsing");
    for (int slots : {1, 2, 4, 8}) for (int stride : {8, 12, 16})
        for (int active = 0; active <= slots; ++active) for (int owner = -1; owner <= slots; ++owner)
            for (int rows = 0; rows <= 16; ++rows) for (bool images : {false, true}) for (bool prefilling : {false, true}) {
                const bool valid = slots == 4 && stride == 12 && active == 1 && owner >= 0 && owner < 4 &&
                                   rows > 0 && rows <= 8 && !images && !prefilling;
                require(slot_solo_eligible(true, slots, stride, active, owner, rows, images, prefilling) == valid, "bounded eligible geometry");
                require(!slot_solo_eligible(false, slots, stride, active, owner, rows, images, prefilling), "off never dispatches solo");
            }
    require(solo_commit_rows(12) == 8 && solo_commit_rows(16) == 8 && solo_commit_rows(6) == 6,
            "logical cap never changes physical stride");
    // Production uses this predicate before launching its common row sampler.
    const float bias = 1;
    for (bool greedy : {false, true}) for (float temp : {0.0f, 0.5f}) for (int history : {0, 64}) {
        require(row_needs_sampling(greedy, temp, history, &bias), "bias must not be skipped on greedy rows");
        require(row_needs_sampling(greedy, temp, history, nullptr) ==
                ((!greedy && temp > 0) || history > 0), "sampled and penalized draws remain authoritative");
    }
    using Bank = SoloGraphBank<uintptr_t>;
    static_assert(!std::is_copy_constructible_v<Bank> && !std::is_copy_assignable_v<Bank>);
    Bank parked[4];
    uintptr_t active[9] = {}, active_nr[9] = {}, commit = 99;
    int groups[9] = {};
    for (int t = 1; t <= 8; ++t) { active[t] = t; active_nr[t] = t + 10; groups[t] = t % 2 + 1; }
    for (int b = 0; b < 4; ++b) {
        parked[b].commit = 1000 + b;
        for (int t = 1; t <= 8; ++t) {
            parked[b].exec[t] = 100 + b * 20 + t;
            parked[b].exec_nr[t] = 110 + b * 20 + t;
            parked[b].groups[t] = 1;
        }
    }
    auto ownership = [&] {
        std::set<uintptr_t> seen;
        auto insert = [&](uintptr_t e) { if (e) require(seen.insert(e).second, "each handle has exactly one owner"); };
        insert(commit);
        for (int t = 1; t <= 8; ++t) { insert(active[t]); insert(active_nr[t]); }
        for (const auto& bank : parked) {
            insert(bank.commit);
            for (int t = 1; t <= 8; ++t) { insert(bank.exec[t]); insert(bank.exec_nr[t]); }
        }
        require(seen.size() == 85, "no handles lost in binding or restoration");
    };
    ownership();
    for (int owner : {3, 1, 0, 2, 2, 3, 0, 1}) {
        parked[owner].exchange(active, active_nr, commit, groups);
        ownership();
        require(commit == (uintptr_t) (1000 + owner), "nonzero owner has its own commit graph");
        for (int t = 1; t <= 8; ++t) {
            require(active[t] == (uintptr_t) (100 + owner * 20 + t), "arbitrary owner target graph");
            require(active_nr[t] == (uintptr_t) (110 + owner * 20 + t), "owner residency variant graph");
            require(groups[t] == 1, "groups exchanged with their target graph");
        }
        parked[owner].exchange(active, active_nr, commit, groups);
        ownership();
        require(commit == 99, "anchor commit restored after accepted-prefix scope");
        for (int t = 1; t <= 8; ++t)
            require(active[t] == (uintptr_t)t && active_nr[t] == (uintptr_t)(t + 10) && groups[t] == t % 2 + 1,
                    "anchor graph/groups restored exactly");
    }
    std::printf("PASS %d Solo dispatch/sampling-predicate/handle ownership checks (CPU only)\n", checks);
}
