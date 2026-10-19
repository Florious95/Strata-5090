// CPU contract for a wide verifier's bounded, group-private IL scratch.
#include "strata/core/batch_segments.hpp"
#include <cstdio>
#include <cstdlib>

int main() {
    using namespace strata::core;
    int checks = 0;
    auto check = [&](bool ok) {
        ++checks;
        if (!ok) { std::fprintf(stderr, "batch_scratch_cpu: FAIL %d\n", checks); std::exit(1); }
    };
    for (int rows = 1; rows <= kBatchMaxRows; ++rows) {
        const int capacity = batch_il_scratch_rows(rows);
        check(capacity >= 4 && capacity <= 8);
        for (int first = 1; first < rows; ++first) {
            int slots[kBatchMaxRows] = {};
            for (int i = first; i < rows; ++i) slots[i] = 1;
            const int cut = segment_group_boundary(slots, rows, true);
            if (cut <= 0) continue; // unsupported layouts use the scalar/refusal path
            check(cut <= capacity && rows - cut <= capacity);
            // Two images never overlap; the largest valid group exactly fits its image.
            const int stride = capacity * 36;
            check(cut * 36 <= stride && stride + (rows - cut) * 36 <= 2 * stride);
            check(qsa_same_slot_tile(slots, rows, 0, cut, 0, true) == (cut >= 2));
            check(qsa_same_slot_tile(slots, rows, cut, rows, 0, true) == (rows - cut >= 2));
        }
    }
    check(batch_il_scratch_rows(12) == 8);
    check(batch_il_scratch_rows(16) == 8);
    std::printf("batch_scratch_cpu: %d checks PASS\n", checks);
}
