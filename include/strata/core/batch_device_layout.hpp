#pragma once

#include "strata/core/batch_segments.hpp"
#include "strata/kernels/batch_device_table.hpp"
#include <array>
#include <climits>
#include <string>

namespace strata::core {
using DeviceWindow = strata::kernels::DeviceWindowLayout;
using DeviceCommit = strata::kernels::DeviceCommitLayout;

// No slot IDs, segment count, generations or keep values are graph-key data.
// profile contains only frozen execution choices; table replacement invalidates
// the entire family outside this key. This is rows_cut, not rows-only.
struct BatchRowsCutKey {
    uint32_t rows = 0, cut = 0, profile = 0;
    bool operator==(const BatchRowsCutKey&) const = default;
    bool operator<(const BatchRowsCutKey& b) const {
        if (rows != b.rows) return rows < b.rows;
        if (cut != b.cut) return cut < b.cut;
        return profile < b.profile;
    }
};

inline bool make_device_window(const int* ids, const int64_t* positions, int rows, int cut,
                               const std::array<uint64_t, 4>& generations,
                               const std::array<uint64_t, 4>& owner_epochs,
                               uint64_t sequence, uint64_t table_epoch, uint32_t profile,
                               DeviceWindow& out, std::string& err) {
    out = {}; // also invalidates every unused row/segment, never preserve stale owners
    if (!sequence || !table_epoch || rows < 1 || rows > 12 || cut < 0 || cut >= rows ||
        (cut == 0 ? rows > 8 : cut > 8 || rows - cut > 8)) {
        err = "device window: invalid sequence/epoch or dense bounds"; return false;
    }
    if (!validate_segments(ids, positions, rows, 12, 4, true, err)) return false;
    if (cut > 0 && ids[cut - 1] == ids[cut]) {
        err = "device window: dense cut bisects a causal segment"; return false;
    }
    out.rows = (uint32_t) rows; out.dense_cut = (uint32_t) cut; out.profile = profile;
    out.sequence = sequence; out.table_epoch = table_epoch;
    for (int first = 0; first < rows;) {
        const int owner = ids[first], length = segment_width(ids, rows, first);
        if (out.segments >= 4 || length > 8 || !generations[owner] || !owner_epochs[owner]) {
            err = "device window: invalid segment capacity or owner identity";
            out = {}; return false;
        }
        auto& s = out.segment[out.segments];
        s.slot = owner; s.first = first; s.length = length;
        s.request_generation = generations[owner]; s.owner_address_epoch = owner_epochs[owner];
        for (int j = 0; j < length; ++j) {
            auto& row = out.row[first + j];
            row.slot = owner; row.segment = (int32_t) out.segments; row.local_row = j;
        }
        ++out.segments;
        first += length;
    }
    return true;
}

inline bool make_device_commit(const DeviceWindow& w, const int* keeps_by_slot,
                               DeviceCommit& out, std::string& err) {
    out = {};
    if (!keeps_by_slot || w.segments < 1 || w.segments > 4 || !w.sequence || !w.table_epoch ||
        w.rows < 1 || w.rows > 12 || w.dense_cut >= w.rows ||
        (w.dense_cut == 0 ? w.rows > 8 : w.dense_cut > 8 || w.rows - w.dense_cut > 8)) {
        err = "device commit: invalid window"; return false;
    }
    bool active[4] = {};
    int covered = 0;
    for (uint32_t i = 0; i < w.segments; ++i) {
        const auto& s = w.segment[i];
        if (s.slot < 0 || s.slot >= 4 || active[s.slot] || s.first != covered ||
            s.length < 1 || s.length > 8 || covered + s.length > (int)w.rows ||
            !s.request_generation || !s.owner_address_epoch ||
            (w.dense_cut > (uint32_t)s.first && w.dense_cut < (uint32_t)(s.first + s.length)) ||
            keeps_by_slot[s.slot] < 1 || keeps_by_slot[s.slot] > s.length) {
            err = "device commit: accepted prefix outside its owner segment";
            out = {}; return false;
        }
        for (int j = 0; j < s.length; ++j) {
            const auto& r = w.row[s.first + j];
            if (r.slot != s.slot || r.segment != (int32_t)i || r.local_row != j) {
                err = "device commit: invalid row mapping"; out = {}; return false;
            }
        }
        active[s.slot] = true;
        covered += s.length;
        out.keep_by_segment[i] = keeps_by_slot[s.slot];
    }
    for (int b = 0; b < 4; ++b) if (!active[b] && keeps_by_slot[b] != 0) {
        err = "device commit: nonzero keep for inactive owner"; out = {}; return false;
    }
    if (w.rows < 1 || w.rows > 12 || covered != (int)w.rows) {
        err = "device commit: uncovered rows"; out = {}; return false;
    }
    out.sequence = w.sequence; out.table_epoch = w.table_epoch;
    return true;
}

// Validate the frozen C4/Q4/20K-resident table before publishing any pointer to
// a graph. Equal-base checks are necessary, not a proof of allocation ranges.
inline bool validate_device_slot_table(const strata::kernels::DeviceSlotTable& t, std::string& err) {
    using namespace strata::kernels;
    std::array<uintptr_t, 4 * 12 * 16 + 8> mutable_bases{};
    size_t count = 0;
    auto independent = [&](const void* p) {
        if (!p) return false;
        const auto value = reinterpret_cast<uintptr_t>(p);
        for (size_t i = 0; i < count; ++i) if (mutable_bases[i] == value) return false;
        mutable_bases[count++] = value;
        return true;
    };
    for (int b = 0; b < 4; ++b) {
        const auto& owner = t.target[b];
        if (!owner.address_epoch || owner.max_cells < 262144 || owner.max_cells > INT32_MAX ||
            owner.gdn_ord0 != 0 || owner.qsa_ord0 != 0 || !independent(owner.gdn_base) ||
            !independent(owner.ple_hist)) {
            err = "device table: target capacity/epoch or mutable base alias"; return false;
        }
    }
    for (int qi = 0; qi < 12; ++qi) for (int b = 0; b < 4; ++b) {
        const auto& q = t.qsa[qi][b];
        const auto& m = q.map;
        if (q.flags != (kDeviceQsaQ4 | kDeviceQsaRotated | kDeviceQsaStreamed) ||
            q.address_epoch != t.target[b].address_epoch || q.max_cells != t.target[b].max_cells ||
            !q.cos_tab || !q.sin_tab || !q.idx_dead || !q.host.k_q4 || !q.host.v_q4 ||
            !m.page_table || !m.slot_block || !m.slot_stamp || !m.slot_ref || !m.ctl ||
            !m.miss_block || !m.miss_slot || q.pools.page_table != m.page_table ||
            m.n_slots != 5120 || m.n_blocks != (q.max_cells + 3) / 4 ||
            !kv_stream_union_fits(m.n_slots, 8, 2051, 4)) {
            err = "device table: incomplete Q4 Host/map/indexer/capacity/epoch"; return false;
        }
        const void* bases[] = {q.pools.k_q4, q.pools.v_q4, q.pools.page_table,
            q.host.k_q4, q.host.v_q4, m.slot_block, m.slot_stamp, m.slot_ref, m.ctl,
            m.miss_block, m.miss_slot, q.idx_tail, q.idx_dead, q.idx_pooled, q.idx_block_pos, q.tail_snapshot};
        for (const auto* p : bases) if (!independent(p)) {
            err = "device table: null or aliased mutable QSA base"; return false;
        }
    }
    return true;
}

inline BatchRowsCutKey rows_cut_key(const DeviceWindow& w) {
    return {w.rows, w.dense_cut, w.profile};
}
} // namespace strata::core
