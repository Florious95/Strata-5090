// Actual MTP host preflight/scratch, with opaque owners; no CUDA or benchmark test implementation.
#include "strata/core/mtp_batch.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
    using namespace strata::core;
    int checks = 0;
    auto check = [&](bool ok) {
        ++checks;
        if (!ok) { std::fprintf(stderr, "mtp_batch_cpu: FAIL %d\n", checks); std::exit(1); }
    };
    struct Round { int* drafter; };
    int owners[9]{};
    check(kMtpBatchMaxSlots == 8);
    for (unsigned mask = 0; mask < 256; ++mask) {
        std::vector<Round> rounds;
        for (int i = 0; i < 8; ++i) if (mask & (1u << i)) rounds.push_back({&owners[i]});
        check(mtp_batch_membership_error(rounds) == nullptr);
        std::reverse(rounds.begin(), rounds.end());
        check(mtp_batch_membership_error(rounds) == nullptr); // active IDs need not be a leading prefix
        for (std::size_t i = 0; i < rounds.size(); ++i) {
            auto nulls = rounds; nulls[i].drafter = nullptr;
            const char* why = mtp_batch_membership_error(nulls);
            check(why && std::strcmp(why, i == 0 ? "mtp batch: invalid membership" : "mtp batch: invalid round") == 0);
            for (std::size_t j = 0; j < i; ++j) {
                auto duplicate = rounds; duplicate[i].drafter = duplicate[j].drafter;
                why = mtp_batch_membership_error(duplicate);
                check(why && std::strcmp(why, "mtp batch: duplicate drafter") == 0);
            }
        }
        // Reusing the same owners in a later window is valid: membership is call-local.
        check(mtp_batch_membership_error(rounds) == nullptr);
    }
    std::vector<Round> too_many;
    for (auto& owner : owners) too_many.push_back({&owner});
    check(mtp_batch_membership_error(too_many) != nullptr);
    for (int generation = 0; generation < 32; ++generation) {
        struct Guarded { uint64_t before = 0x12345678; MtpBatchScratch scratch; uint64_t after = 0x87654321; } g;
        for (std::size_t i = 0; i < 8; ++i) {
            check(g.scratch.limits[i] == 0 && g.scratch.counts[i] == 0 && !g.scratch.active[i]);
            check(g.scratch.prev[i][0] == 0 && g.scratch.prev[i][1] == 0);
            g.scratch.limits[i] = int(i) + 1;
            g.scratch.counts[i] = generation;
            g.scratch.prev[i] = {int32_t(i), generation};
            g.scratch.active[i] = true;
        }
        g.scratch.active.fill(false); // each draft step clears all eight, not only the first four
        for (std::size_t i = 0; i < 8; ++i) {
            check(!g.scratch.active[i]);
            check(g.scratch.limits[i] == int(i) + 1 && g.scratch.counts[i] == generation);
            check(g.scratch.prev[i][0] == int32_t(i) && g.scratch.prev[i][1] == generation);
        }
        check(g.before == 0x12345678 && g.after == 0x87654321);
    }
    std::printf("mtp_batch_cpu: %d checks PASS\n", checks);
}
