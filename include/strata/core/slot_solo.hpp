#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <utility>

namespace strata::core {

inline bool parse_slot_solo(const char* text, bool& enabled) {
    enabled = text && text[0] == '1' && text[1] == '\0';
    return !text || ((text[0] == '0' || text[0] == '1') && text[1] == '\0');
}

// Explicit GRAPH_LIMIT=8 must reach the Solo LRU as well as the legacy one.
// Keep the configured default (4); cap explicit exploration at eight targets.
inline constexpr std::size_t slot_solo_graph_limit(std::size_t limit) {
    return std::clamp<std::size_t>(limit, 1, 8);
}

// Logical commit width is not the verifier's per-layer physical row stride.
inline constexpr int solo_commit_rows(int physical_rows) { return std::min(physical_rows, 8); }

inline bool slot_solo_eligible(bool enabled, int slots, int physical_rows, int active,
                               int owner, int rows, bool has_images, bool prefilling) {
    return enabled && slots == 4 && physical_rows == 12 && active == 1 && owner >= 0 && owner < slots &&
           rows >= 1 && rows <= solo_commit_rows(physical_rows) && !has_images && !prefilling;
}

inline bool row_needs_sampling(bool greedy, float temperature, int history, const float* bias) {
    return (!greedy && temperature > 0.0f) || history > 0 || bias != nullptr;
}

// Only graph handles move; sequence state and the verifier's scratch never do.
// The owning Verifier drains all streams before exchange/destruction. No copying
// is permitted (including between the active bank and a slot's parked bank).
template<class Handle> struct SoloGraphBank {
    Handle exec[9] = {}, exec_nr[9] = {}, commit = {};
    int groups[9] = {};
    SoloGraphBank() = default;
    SoloGraphBank(const SoloGraphBank&) = delete;
    SoloGraphBank& operator=(const SoloGraphBank&) = delete;
    void exchange(Handle (&a)[9], Handle (&b)[9], Handle& c, int (&g)[9]) {
        std::swap(exec, a); std::swap(exec_nr, b); std::swap(commit, c); std::swap(groups, g);
    }
};

} // namespace strata::core
