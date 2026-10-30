// Host-only Solo dispatch, logical-cap and graph-handle ownership contracts.
// Does NOT execute CUDA, attention, GDN, sampling kernels or SessionState binding.
#include "strata/core/slot_solo.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
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
    for (std::size_t configured : {0u, 1u, 2u, 4u, 5u, 6u, 7u, 8u, 64u})
        require(slot_solo_graph_limit(configured) == (configured < 1 ? 1 : configured > 8 ? 8 : configured),
                "configured Solo limit reaches eight without changing default four");
    std::size_t solo_limit = 0;
    require(parse_solo_graph_limit(nullptr, 4, solo_limit) && solo_limit == 4, "unset Solo limit follows the default four");
    require(parse_solo_graph_limit(nullptr, 64, solo_limit) && solo_limit == 8, "unset Solo limit follows GRAPH_LIMIT up to eight");
    for (int value = 1; value <= 16; ++value) {
        const std::string text = std::to_string(value);
        require(parse_solo_graph_limit(text.c_str(), 4, solo_limit) && solo_limit == (std::size_t) value,
                "explicit Solo limit 1..16");
    }
    for (const char* invalid : {"", "0", "17", "64", "-1", "8 ", "x"})
        require(!parse_solo_graph_limit(invalid, 4, solo_limit), "strict Solo limit parsing");
    for (int slots : {1, 2, 4, 8}) for (int stride : {8, 12, 16})
        require(slot_solo_geometry(slots, stride) == ((slots == 4 && stride == 12) || (slots == 8 && stride == 16)),
                "Solo geometry is C4 rows12 or C8 rows16");
    require(kSlotSoloMaxOwners == 8, "Solo banks cover C8");
    for (int slots : {1, 2, 4, 8}) for (int stride : {8, 12, 16})
        for (int active = 0; active <= slots; ++active) for (int owner = -1; owner <= slots; ++owner)
            for (int rows = 0; rows <= 16; ++rows) for (bool images : {false, true}) for (bool prefilling : {false, true}) {
                const bool valid = ((slots == 4 && stride == 12) || (slots == 8 && stride == 16)) && active == 1 &&
                                   owner >= 0 && owner < slots && rows > 0 && rows <= 8 && !images && !prefilling;
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
    Bank parked[kSlotSoloMaxOwners];
    uintptr_t active[9] = {}, active_nr[9] = {}, commit = 99;
    int groups[9] = {};
    bool launched[9] = {}, launched_nr[9] = {};
    for (int t = 1; t <= 8; ++t) {
        active[t] = t; active_nr[t] = t + 10; groups[t] = t % 2 + 1;
        launched[t] = t % 3 == 0; launched_nr[t] = t % 3 == 1;
    }
    for (int b = 0; b < kSlotSoloMaxOwners; ++b) {
        parked[b].commit = 1000 + b;
        for (int t = 1; t <= 8; ++t) {
            parked[b].exec[t] = 100 + b * 20 + t;
            parked[b].exec_nr[t] = 110 + b * 20 + t;
            parked[b].groups[t] = 1;
            parked[b].launched[t] = t == b + 1;   // a parked graph keeps its own first-launch state
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
        require(seen.size() == 1 + 16 + kSlotSoloMaxOwners * 17, "no handles lost in binding or restoration");
    };
    ownership();
    for (int owner : {3, 1, 0, 2, 2, 3, 0, 1, 7, 4, 6, 5, 5, 7}) {
        parked[owner].exchange(active, active_nr, commit, groups, launched, launched_nr);
        ownership();
        require(commit == (uintptr_t) (1000 + owner), "nonzero owner has its own commit graph");
        for (int t = 1; t <= 8; ++t) {
            require(active[t] == (uintptr_t) (100 + owner * 20 + t), "arbitrary owner target graph");
            require(active_nr[t] == (uintptr_t) (110 + owner * 20 + t), "owner residency variant graph");
            require(groups[t] == 1, "groups exchanged with their target graph");
            require(launched[t] == (t == owner + 1) && !launched_nr[t], "first-launch flags exchanged with their graphs");
        }
        parked[owner].exchange(active, active_nr, commit, groups, launched, launched_nr);
        ownership();
        require(commit == 99, "anchor commit restored after accepted-prefix scope");
        for (int t = 1; t <= 8; ++t)
            require(active[t] == (uintptr_t)t && active_nr[t] == (uintptr_t)(t + 10) && groups[t] == t % 2 + 1 &&
                    launched[t] == (t % 3 == 0) && launched_nr[t] == (t % 3 == 1),
                    "anchor graph/groups/first-launch flags restored exactly");
    }
    {   // 0.2.8 wide build: the verifier's arrays have kWindowMaxT + 1 = 17 entries; the bank swaps [0..8] only
        Bank bank;
        uintptr_t w[17] = {}, w_nr[17] = {}, w_commit = 7;
        int w_groups[17] = {};
        bool w_la[17] = {}, w_lb[17] = {};
        for (int t = 1; t <= 8; ++t) { bank.exec[t] = 500 + t; bank.exec_nr[t] = 600 + t; bank.groups[t] = 2; bank.launched[t] = true; }
        bank.commit = 700;
        for (int t = 1; t <= 16; ++t) { w[t] = 800 + t; w_nr[t] = 900 + t; w_groups[t] = 1; w_lb[t] = true; }
        require(!Bank::narrow_only(w) && !Bank::narrow_only(w_nr), "a held 9..16-row graph is seen (the binding refuses it)");
        bank.exchange(w, w_nr, w_commit, w_groups, w_la, w_lb);
        for (int t = 1; t <= 8; ++t)
            require(w[t] == (uintptr_t) (500 + t) && w_nr[t] == (uintptr_t) (600 + t) && w_groups[t] == 2 && w_la[t] && !w_lb[t] &&
                    bank.exec[t] == (uintptr_t) (800 + t) && bank.exec_nr[t] == (uintptr_t) (900 + t) && bank.groups[t] == 1 &&
                    !bank.launched[t] && bank.launched_nr[t], "the narrow graphs, groups and flags change places");
        for (int t = 9; t <= 16; ++t)
            require(w[t] == (uintptr_t) (800 + t) && w_nr[t] == (uintptr_t) (900 + t) && w_groups[t] == 1 && !w_la[t] && w_lb[t],
                    "the wide entries stay with the verifier");
        require(w_commit == 700 && bank.commit == 7, "the commit graph changes places");
        for (int t = 9; t <= 16; ++t) w[t] = w_nr[t] = 0;
        require(Bank::narrow_only(w) && Bank::narrow_only(w_nr), "no wide graph held: the binding may proceed");
        uintptr_t n9[9] = {1, 2};
        require(Bank::narrow_only(n9), "the 8-row build's arrays are always narrow");
    }
    std::printf("PASS %d Solo dispatch/sampling-predicate/handle ownership checks (CPU only)\n", checks);
}
