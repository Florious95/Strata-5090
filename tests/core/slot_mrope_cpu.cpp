// CPU-only tests of the production position helper and capture-pointer scopes.
#include "strata/kernels/mrope.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

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
    // C4 Row Table/Solo routing: a null table with cell positions equals a bank only when every
    // cell of it is (c, c, c). Slots 0/1 hold the images above, 2/3 text, 4 a 1x6 strip (delta 0).
    std::array<std::array<int32_t, 128 * 3>, 5> c4{};
    c4[0] = banks[0]; c4[1] = banks[1];
    for (int b = 2; b < 5; ++b) for (int c = 0; c < 128; ++c) for (int s = 0; s < 3; ++s) c4[b][c * 3 + s] = c;
    for (int c = 4; c < 10; ++c) { c4[4][c * 3] = c4[4][c * 3 + 1] = 4; c4[4][c * 3 + 2] = c; }
    const bool identity[5] = {false, false, true, true, false};   // generate.cpp: no picture in the prompt
    const int32_t c4_delta[5] = {delta[0], delta[1], 0, 0, 0}, c4_end[5] = {10, 16, 20, 20, 10};
    bool text_only[5];
    for (int b = 0; b < 5; ++b) {
        bool exact = true;
        for (int c = 0; c < 128; ++c) {
            for (int s = 0; s < 3; ++s) exact = exact && c4[b][c * 3 + s] == c;
            exact = exact && k::mrope_text_position(c, c4_end[b], c4_delta[b], c4[b].data()) == c;
        }
        text_only[b] = k::mrope_text_only(c4[b].data(), identity[b], c4_delta[b]);
        check(text_only[b] == exact);
        check(text_only[b] == (b == 2 || b == 3));
        check(!k::mrope_text_only(c4[b].data(), identity[b], -3)); // a restored nonzero delta never qualifies
    }
    check(k::mrope_text_only(nullptr, false, -3)); // no --vision: no bank, the text path is unchanged
    auto table_window = [&](std::initializer_list<int> rows) {
        bool ok = true;
        for (int r : rows) ok = ok && text_only[r];
        return ok;
    };
    check(table_window({2, 2, 3, 3, 3}) && table_window({3}));
    check(!table_window({2, 0, 0, 3}) && !table_window({1}) && !table_window({3, 3, 4}));
    // C8-4 Vision8: generate.cpp's table of max(batch, 2) banks, slot b at mrope_bank_offset(b, cells). The banks are
    // disjoint and tile the table for 2/4/8 slots, also at the production 262144 + 64 cells.
    for (int64_t cells : {int64_t(128), int64_t(262208)})
        for (int slots : {2, 4, 8}) {
            check(k::mrope_bank_offset(0, cells) == 0);
            for (int b = 0; b < slots; ++b)
                check(k::mrope_bank_offset(b + 1, cells) - k::mrope_bank_offset(b, cells) == size_t(cells) * 3);
            check(k::mrope_bank_offset(slots, cells) == size_t(slots) * size_t(cells) * 3);
        }
    // Owners 0..7, each its own image (or none) and delta. An admission rewrites only its slot's bank (generate.cpp's
    // put/upload_mrope over [offset, offset + cells * 3)); every other bank keeps its bytes.
    constexpr int C8 = 8, CELLS = 128;
    std::array<int32_t, C8 * CELLS * 3> table{};
    auto bank = [&](int b) { return table.data() + k::mrope_bank_offset(b, CELLS); };
    for (int b = 0; b < C8; ++b) for (int c = 0; c < CELLS; ++c) for (int s = 0; s < 3; ++s) bank(b)[c * 3 + s] = c;
    const int img_w[C8] = {2, 0, 3, 0, 4, 0, 1, 5}, img_h[C8] = {3, 0, 4, 0, 2, 0, 6, 1};   // 0: a text slot
    int32_t c8_delta[C8]{}, c8_end[C8]{};
    bool c8_identity[C8]{};
    for (int b = 0; b < C8; ++b) {
        const std::array<int32_t, C8 * CELLS * 3> before = table;
        const int start = 4 + b, n = img_w[b] * img_h[b], end = start + n;
        c8_identity[b] = n == 0;
        c8_delta[b] = n ? std::max(img_w[b], img_h[b]) - n : 0;
        c8_end[b] = n ? end : 20 + b;
        for (int c = start; c < end; ++c) {
            bank(b)[c * 3] = start;
            bank(b)[c * 3 + 1] = start + (c - start) / img_w[b];
            bank(b)[c * 3 + 2] = start + (c - start) % img_w[b];
        }
        for (int c = end; n && c < CELLS; ++c) for (int s = 0; s < 3; ++s) bank(b)[c * 3 + s] = c + c8_delta[b];
        const size_t lo = k::mrope_bank_offset(b, CELLS), hi = k::mrope_bank_offset(b + 1, CELLS);
        for (size_t i = 0; i < table.size(); ++i) check(i >= lo && i < hi ? true : table[i] == before[i]);
    }
    bool c8_text[C8];
    for (int b = 0; b < C8; ++b) {
        c8_text[b] = k::mrope_text_only(bank(b), c8_identity[b], c8_delta[b]);
        check(c8_text[b] == (img_w[b] == 0));
        // the row's own bank and delta: a decode row of owner b at cell c rotates at c + delta[b], never another's
        for (int c = c8_end[b]; c < CELLS; ++c) {
            check(k::mrope_text_position(c, c8_end[b], c8_delta[b], bank(b)) == c + c8_delta[b]);
            check(k::mrope_text_position(c, c8_end[b], c8_delta[b], bank(b)) == bank(b)[c * 3]);
        }
        for (int c = 0; c < c8_end[b]; ++c) check(k::mrope_text_position(c, c8_end[b], c8_delta[b], bank(b)) == bank(b)[c * 3]);
    }
    check(c8_delta[6] == 0 && !c8_text[6]);   // a 1x6 strip moves no delta and is still an image owner
    // verify.cpp stage_batch: a window stays on the row table only when every row's owner is text (1, 3, 5)
    auto c8_window = [&](std::initializer_list<int> rows) {
        bool ok = true;
        for (int r : rows) ok = ok && c8_text[r];
        return ok;
    };
    check(c8_window({1, 1, 3, 3, 5, 5}) && c8_window({5}));
    check(!c8_window({1, 3, 4, 5}) && !c8_window({7}) && !c8_window({0, 1, 2, 3, 4, 5, 6, 7}));
    // owners 4..7 under one capture scope each: the scope is the row's bank and restores the previous table
    current = nullptr;
    for (int b = 4; b < C8; ++b) {
        k::MropeTableScope own(bank(b)); check(current == bank(b));
        { k::MropeTableScope merged(nullptr); check(current == nullptr); }
        check(current == bank(b));
    }
    check(current == nullptr);
    std::printf("slot_mrope_cpu: %d checks PASS; layouts=1,7,8,9,10,11,12; c4_routing=5; c8_banks=8\n",checks);
}
