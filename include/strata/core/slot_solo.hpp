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

// C8-2: STRATA_BATCH_SOLO_GRAPH_LIMIT sets the Solo target LRU on its own (1..16;
// C8 has twice the owners). Unset keeps slot_solo_graph_limit(batch limit).
inline bool parse_solo_graph_limit(const char* text, std::size_t batch_limit, std::size_t& out) {
    out = slot_solo_graph_limit(batch_limit);
    if (!text) return true;
    if (!*text) return false;
    std::size_t value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        value = value * 10 + (std::size_t) (*p - '0');
        if (value > 16) return false;
    }
    if (value < 1) return false;
    out = value;
    return true;
}

// Logical commit width is not the verifier's per-layer physical row stride.
inline constexpr int solo_commit_rows(int physical_rows) { return std::min(physical_rows, 8); }

// Solo owners: C4 rows12 (accepted) and C8 rows16 (C8-2, exploratory). Both are
// two eight-row groups at most; a Solo window is one owner's <=8 rows either way.
inline constexpr int kSlotSoloMaxOwners = 8;
inline constexpr bool slot_solo_geometry(int slots, int physical_rows) {
    return (slots == 4 && physical_rows == 12) || (slots == 8 && physical_rows == 16);
}

inline bool slot_solo_eligible(bool enabled, int slots, int physical_rows, int active,
                               int owner, int rows, bool has_images, bool prefilling) {
    return enabled && slot_solo_geometry(slots, physical_rows) && active == 1 && owner >= 0 && owner < slots &&
           rows >= 1 && rows <= solo_commit_rows(physical_rows) && !has_images && !prefilling;
}

inline bool row_needs_sampling(bool greedy, float temperature, int history, const float* bias) {
    return (!greedy && temperature > 0.0f) || history > 0 || bias != nullptr;
}

// Only graph handles move; sequence state and the verifier's scratch never do.
// The owning Verifier drains all streams before exchange/destruction. No copying
// is permitted (including between the active bank and a slot's parked bank).
// The verifier's one-CTA select row only (sel_long_ 0): a Solo binding refuses STRATA_SELECT_LONG. A graph's
// first-launch flags move with it, so a parked graph is never mistaken for a launched one.
template<class Handle> struct SoloGraphBank {
    Handle exec[9] = {}, exec_nr[9] = {}, commit = {};
    int groups[9] = {};
    bool launched[9] = {}, launched_nr[9] = {};
    SoloGraphBank() = default;
    SoloGraphBank(const SoloGraphBank&) = delete;
    SoloGraphBank& operator=(const SoloGraphBank&) = delete;
    void exchange(Handle (&a)[9], Handle (&b)[9], Handle& c, int (&g)[9], bool (&la)[9], bool (&lb)[9]) {
        std::swap(exec, a); std::swap(exec_nr, b); std::swap(commit, c); std::swap(groups, g);
        std::swap(launched, la); std::swap(launched_nr, lb);
    }
};

} // namespace strata::core
