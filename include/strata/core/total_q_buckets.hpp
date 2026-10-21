#pragma once

#include "strata/core/batch_segments.hpp"
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace strata::core {

// Four families, EIGHT exact G2 templates. A family is not one padded graph:
// actual rows and causal cut remain key data. No dummy state owners are used.
inline constexpr int kTotalQTemplateKeyTag = -3;
struct TotalQTemplate {
    int bucket, rows, cut;
    std::array<int, 4> witness;
};
inline constexpr std::array<TotalQTemplate, 8> kTotalQTemplates{{
    {3, 2, 1, {1, 1, 0, 0}},
    {3, 3, 1, {1, 2, 0, 0}},
    {3, 3, 2, {2, 1, 0, 0}},
    {6, 4, 2, {1, 1, 1, 1}},
    {6, 6, 3, {3, 3, 0, 0}},
    {9, 9, 6, {3, 3, 3, 0}},
    {12, 12, 6, {6, 6, 0, 0}},
    {12, 12, 8, {4, 4, 4, 0}},
}};

inline constexpr int total_q_template_id(int rows, int cut) {
    for (int i = 0; i < (int)kTotalQTemplates.size(); ++i)
        if (kTotalQTemplates[i].rows == rows && kTotalQTemplates[i].cut == cut) return i;
    return -1;
}
inline constexpr std::array<int, 4> total_q_key_prefix(const TotalQTemplate& t) {
    return {kTotalQTemplateKeyTag, t.bucket, t.rows, t.cut};
}
inline bool total_q_pinned_key(const std::vector<int>& key) {
    if (key.size() < 4 || key[0] != kTotalQTemplateKeyTag) return false;
    const int id = total_q_template_id(key[2],key[3]);
    return id >= 0 && kTotalQTemplates[id].bucket == key[1];
}
template<class GraphMap> inline size_t total_q_evictable_count(const GraphMap& graphs) {
    size_t count = 0;
    for (const auto& entry : graphs) count += !total_q_pinned_key(entry.first);
    return count;
}

struct TotalQPlan {
    std::array<int, 4> widths{};
    int template_id = -1;
};

// Post-filter only: never grow a proposal prefix, remove a pending input, move
// a canonical owner or change within-owner order. The caller keeps Solo and
// admission outside this policy. All profiles must retain G2 dense parallelism.
inline bool plan_total_q_buckets(const std::array<int, 4>& input, int first,
                                 TotalQPlan& out, std::string& error) {
    const auto wanted = input; // also safe when input aliases out.widths
    out = {};
    error.clear();
    int active = 0, total = 0;
    for (const int n : wanted) {
        if (n < 0 || n > 8) { error = "total-Q buckets: invalid per-owner width"; return false; }
        active += n > 0; total += n;
    }
    if (first < 0 || first >= 4 || active < 2 || total > 12) {
        error = "total-Q buckets: requires 2..4 owners within twelve rows"; return false;
    }
    int best_rows = 0;
    auto preferred = [&](const std::array<int, 4>& candidate) {
        for (int j = 0; j < 4; ++j) {
            const int b = (first + j) % 4;
            if (candidate[b] != out.widths[b]) return candidate[b] > out.widths[b];
        }
        return false;
    };
    // At total <=12, at most 81 prefix vectors are considered. This is not a
    // GPU sort or a replacement for the confidence/cost planner upstream.
    for (int a = wanted[0] > 0; a <= wanted[0]; ++a)
    for (int b = wanted[1] > 0; b <= wanted[1]; ++b)
    for (int c = wanted[2] > 0; c <= wanted[2]; ++c)
    for (int d = wanted[3] > 0; d <= wanted[3]; ++d) {
        const std::array<int, 4> widths{a,b,c,d};
        int rows[12] = {}, n = 0;
        for (int owner = 0; owner < 4; ++owner)
            for (int j = 0; j < widths[owner]; ++j) rows[n++] = owner;
        const int cut = segment_group_boundary(rows,n,true,true);
        const int id = total_q_template_id(n,cut);
        if (id < 0 || n < best_rows || (n == best_rows && !preferred(widths))) continue;
        out.widths = widths; out.template_id = id; best_rows = n;
    }
    // Pending-only widths always reach R2/c1, R3/c1 or R4/c2.
    if (out.template_id < 0) { error = "total-Q buckets: no causal template"; return false; }
    return true;
}
} // namespace strata::core
