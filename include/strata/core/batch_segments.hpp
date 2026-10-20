#pragma once

#include <cstdint>
#include <string>

namespace strata::core {

inline constexpr int kBatchMaxRows = 16;

// Bound graph-pair retention before CUDA OOM. Both audited A/B arms use the
// same small default; ordinary serving retains the existing 64-layout policy.
inline bool parse_batch_graph_limit(const char* text, bool experiment, int& out) {
    out = experiment ? 4 : 64;
    if (!text) return true;
    if (!*text) return false;
    int value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        value = value * 10 + (*p - '0');
        if (value > 64) return false;
    }
    if (value < 1) return false;
    out = value;
    return true;
}

// The native interleaved quantizer accepts 1..8 columns, even when the
// verifier owns more plain row images. Each dense group gets its own image.
inline constexpr int batch_il_scratch_rows(int verifier_rows) {
    return verifier_rows < 4 ? 4 : verifier_rows > 8 ? 8 : verifier_rows;
}

// Give every active request its pending row, then rotate access to spare rows.
// Callers bound legacy widths to four, or C2 widths to six, preserving two <=8-row groups.
inline void allocate_batch_rows(const int* wanted, int count, int budget, int first, int* out) {
    for (int i = 0; i < count; ++i) {
        out[i] = wanted[i] > 0 ? 1 : 0;
        budget -= out[i];
    }
    while (budget > 0) {
        bool changed = false;
        for (int j = 0; j < count && budget > 0; ++j) {
            const int i = (first + j) % count;
            if (out[i] < wanted[i]) { ++out[i]; --budget; changed = true; }
        }
        if (!changed) break;
    }
}

// Expand only fully useful, fully accepted work. Poor VRAM hit rate forbids
// wide windows; partial allocation or the probability gate alone cannot grow it.
inline int adaptive_c2_row_limit(int limit, int maximum, int useful, int kept,
                                 int64_t hits, int64_t routed) {
    if (routed > 0 && hits * 100 < routed * 93) return limit > 4 ? 4 : limit;
    if (useful == limit && kept == useful && limit < maximum) return limit + 1;
    if (useful == limit && kept * 2 < useful && limit > 2) return limit - 1;
    return limit;
}

// Preserve complete causal segments while bounding each execution group to the
// existing eight-row kernels. Zero means no second group, -1 means unsupported.
inline int segment_group_boundary(const int* slots, int count, bool parallel_two_slots = false,
                                  bool parallel_segments = false) {
    if (count <= 8) {
        if (parallel_segments) {
            int best = 0;
            for (int i = 1; i < count; ++i)
                if (slots[i - 1] != slots[i] &&
                    (best == 0 || (i * 2 - count) * (i * 2 - count) < (best * 2 - count) * (best * 2 - count)))
                    best = i;
            return best;   // complete slot segments; two execution groups are not two requests
        }
        if (!parallel_two_slots || count < 2) return 0;
        int cut = 1;
        while (cut < count && slots[cut] == slots[0]) ++cut;
        if (cut == count) return 0;   // never split one slot's causal segment
        for (int i = cut + 1; i < count; ++i)
            if (slots[i] != slots[cut]) return 0;   // only two complete slot tiles
        return cut;
    }
    for (int i = 8; i > 0; --i)
        if (slots[i - 1] != slots[i] && count - i <= 8) return i;
    return -1;
}

// Trim only spare proposal rows to fit two <=8-column dense groups. Preserve
// slot order, one pending input per live slot, and rotate which slot yields a
// spare row. A capacity-valid layout such as [6,6,4] is not itself packable.
// Returns the packed row count, or -1 for invalid input. Inactive widths are 0.
inline int pack_segment_rows(int* widths, int count, int first) {
    if (!widths || count < 1 || count > kBatchMaxRows || first < 0 || first >= count) return -1;
    int total = 0, active = 0;
    for (int i = 0; i < count; ++i) {
        if (widths[i] < 0 || widths[i] > 8) return -1;
        total += widths[i];
        active += widths[i] > 0;
    }
    if (total > kBatchMaxRows || active > 16) return -1;
    if (total <= 8) return total;
    int cut = -1, loss = kBatchMaxRows + 1, left = 0, live_left = 0;
    for (int i = 0; i < count - 1; ++i) {
        left += widths[i];
        live_left += widths[i] > 0;
        if (!live_left || ! (active - live_left) || live_left > 8 || active - live_left > 8) continue;
        const int trim = (left > 8 ? left - 8 : 0) + (total - left > 8 ? total - left - 8 : 0);
        if (trim < loss) { cut = i + 1; loss = trim; }
    }
    if (cut < 0) return -1;
    for (int side = 0; side < 2; ++side) {
        const int begin = side == 0 ? 0 : cut, end = side == 0 ? cut : count;
        int rows = 0;
        for (int i = begin; i < end; ++i) rows += widths[i];
        while (rows > 8) {
            bool changed = false;
            for (int j = 0; j < count && rows > 8; ++j) {
                const int i = (first + j) % count;
                if (i >= begin && i < end && widths[i] > 1) {
                    --widths[i]; --rows; --total; changed = true;
                }
            }
            if (!changed) return -1;
        }
    }
    return total;
}

// The pooled scorer's MQ=8 path can share keys only within a full-Q4,
// non-streamed slot tile. Never merge independent slot states or change append order.
inline bool qsa_same_slot_tile(const int* slots, int count, int first, int end,
                               int kv_mode, bool full_q4) {
    if (!slots || first < 0 || first >= count || end <= first || end > count ||
        end - first < 2 || end - first > 8 ||
        kv_mode != 0 || !full_q4 || slots[first] < 0) return false;
    for (int t = first + 1; t < end; ++t) if (slots[t] != slots[first]) return false;
    return true;
}

// ngram_rows consumes a separate oldest-first pair per row. Its API does not
// advance a single predecessor pair across a sequence.
inline void segment_predecessors(const int32_t* tokens, int width, const int32_t* initial, int32_t* pairs) {
    int32_t older = initial[0], newer = initial[1];
    for (int i = 0; i < width; ++i) {
        pairs[2 * i] = older;
        pairs[2 * i + 1] = newer;
        older = newer;
        newer = tokens[i];
    }
}

inline int accepted_segment_prefix(const int32_t* inputs, const int32_t* outputs, int width,
                                    const int64_t* eos_ids, int eos_count) {
    if (!inputs || !outputs || width < 1) return 0;
    int accepted = 1;
    while (accepted < width && inputs[accepted] == outputs[accepted - 1]) ++accepted;
    for (int i = 0; i < accepted; ++i)
        for (int j = 0; j < eos_count; ++j)
            if (outputs[i] == eos_ids[j]) return i + 1;
    return accepted;
}

// Packed rows of one slot form one contiguous causal segment. Independent
// slots may appear in any order, but a slot may not reappear after its segment.
inline int segment_width(const int* slots, int count, int first) {
    int end = first + 1;
    while (end < count && slots[end] == slots[first]) ++end;
    return end - first;
}

inline bool validate_segments(const int* slots, const int64_t* positions, int count,
                              int capacity, int slot_count, bool allow_multiple,
                              std::string& err) {
    if (!slots || !positions || count < 1 || count > capacity || slot_count < 1) {
        err = "verify: invalid segment buffers or row count";
        return false;
    }
    for (int first = 0; first < count;) {
        const int width = segment_width(slots, count, first);
        if (slots[first] < 0 || slots[first] >= slot_count || (!allow_multiple && width != 1)) {
            err = "verify: invalid slot or multiple rows in ordinary batching";
            return false;
        }
        for (int i = 0; i < first; ++i) if (slots[i] == slots[first]) {
            err = "verify: slot segments must be contiguous";
            return false;
        }
        for (int i = first; i < first + width; ++i) {
            // Positions are uploaded as int32. Compare without signed overflow.
            if (positions[i] < 0 || positions[i] > INT32_MAX ||
                (i > first && positions[i] - positions[i - 1] != 1)) {
                err = "verify: invalid or nonconsecutive segment positions";
                return false;
            }
        }
        first += width;
    }
    return true;
}

// keep[first] is the accepted input prefix; continuation rows must hold zero.
// At least the pending input token is committed. A rejected draft is never
// counted merely because its target output was computed.
inline bool validate_segment_keeps(const int* slots, int count, const int* keep,
                                   std::string& err) {
    if (!slots || !keep || count < 1) { err = "verify: missing segment commits"; return false; }
    for (int first = 0; first < count;) {
        const int width = segment_width(slots, count, first);
        if (keep[first] < 1 || keep[first] > width) {
            err = "verify: accepted prefix outside segment"; return false;
        }
        for (int i = first + 1; i < first + width; ++i) if (keep[i] != 0) {
            err = "verify: commit count belongs at the segment start"; return false;
        }
        first += width;
    }
    return true;
}

} // namespace strata::core
