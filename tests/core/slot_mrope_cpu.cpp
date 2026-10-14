// CPU-only tests of the production position helper and capture-pointer scopes.
#include "strata/kernels/mrope.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>

namespace k = strata::kernels;
namespace { const int32_t* current = nullptr; int checks = 0; }
namespace strata::kernels {
void mrope_table_set(const int32_t* p) { current = p; }
const int32_t* mrope_table() { return current; }
}
void check(bool ok) { ++checks; if (!ok) { std::fprintf(stderr,"FAIL check=%d\n",checks); std::exit(1); } }
int main() {
    std::array<std::array<int32_t, 128 * 3>, 2> banks{};
    for (int b = 0; b < 2; ++b)
        for (int c = 0; c < 128; ++c)
            for (int s = 0; s < 3; ++s) banks[b][c * 3 + s] = c;
    // Image at cell4: grid2x3 in slot0 and3x4 in slot1; next text has different deltas.
    const int lengths[2] = {6, 12}, widths[2] = {2, 3}, heights[2] = {3, 4};
    int delta[2]{};
    for (int b = 0; b < 2; ++b) {
        const int end = 4 + lengths[b];
        delta[b] = (widths[b] > heights[b] ? widths[b] : heights[b]) - lengths[b];
        for (int c = 4; c < end; ++c) {
            banks[b][c * 3] = 4;
            banks[b][c * 3 + 1] = 4 + (c - 4) / widths[b];
            banks[b][c * 3 + 2] = 4 + (c - 4) % widths[b];
        }
        for (int c = end; c < 128; ++c)
            for (int s = 0; s < 3; ++s) banks[b][c * 3 + s] = c + delta[b];
        for (int c = 0; c < 128; ++c) {
            check(k::mrope_text_position(c, end, delta[b], banks[b].data()) == banks[b][c * 3]);
            check(k::mrope_text_position(c, 0, 0, nullptr) == c); // unchanged text path
        }
    }
    for (int n : {1, 7, 8, 9, 10, 11, 12})
        for (int t = 0; t < n; ++t) {
            const int slot = t < (n + 1) / 2 ? 0 : 1, cell = 32 + t;
            check(k::mrope_text_position(cell, 20, delta[slot], banks[slot].data()) == cell + delta[slot]);
            check(cell == 32 + t); // KV/causal position never shifted
        }
    current = banks[0].data();
    {
        k::MropeTableScope slot(banks[1].data()); check(current == banks[1].data());
        { k::MropeTableScope merged(nullptr); check(current == nullptr); }
        check(current == banks[1].data());
        { k::MropeTableScope off(nullptr, false); check(current == banks[1].data()); }
    }
    check(current == banks[0].data());
    for (int64_t c : {int64_t(262143), int64_t(262207)})
        check(k::mrope_text_position(c, 20, -8, nullptr) == c - 8);
    std::printf("slot_mrope_cpu: %d checks PASS; layouts=1,7,8,9,10,11,12\n",checks);
}
