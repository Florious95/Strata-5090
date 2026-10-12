#include "strata/core/batch_segments.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>

int main() {
    using strata::core::qsa_same_slot_tile;
    int checks = 0;
    for (int count = 1; count <= 16; ++count) {
        // Exhaustive binary layouts include mixed/fragmented and both slot orders.
        for (unsigned mask = 0; mask < (1u << count); ++mask) {
            std::array<int, 16> rows{};
            for (int t = 0; t < count; ++t) rows[t] = (mask >> t) & 1u;
            for (int first = 0; first < count; ++first) for (int end = first + 1; end <= count; ++end) {
                bool same = end-first >= 2 && end-first <= 8;
                for (int t = first + 1; t < end; ++t) same &= rows[t] == rows[first];
                if (qsa_same_slot_tile(rows.data(),count,first,end,0,true) != same ||
                    qsa_same_slot_tile(rows.data(),count,first,end,1,true) ||
                    qsa_same_slot_tile(rows.data(),count,first,end,0,false)) return 1;
                ++checks;
            }
        }
    }
    const int rows[] = {0,0,0,1,1,1};
    if (qsa_same_slot_tile(nullptr,6,0,3,0,true) || qsa_same_slot_tile(rows,6,-1,3,0,true) ||
        qsa_same_slot_tile(rows,6,0,7,0,true) || qsa_same_slot_tile(rows,6,4,2,0,true) ||
        qsa_same_slot_tile(rows,6,1,std::numeric_limits<int>::min(),0,true)) return 1;
    const int invalid[] = {-1,-1};
    if (qsa_same_slot_tile(invalid,2,0,2,0,true)) return 1;
    std::printf("PASS: %d exhaustive tile guards; mixed/streamed/non-Q4/single/wide/invalid fallback\n",checks);
}
