#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace strata::core {

// Independent draft owners, not verifier rows or proposals per owner.
inline constexpr std::size_t kMtpBatchMaxSlots = 8;

// CPU-only preflight, before dereferencing a device or capturing/launching graphs.
template<class Rounds>
inline const char* mtp_batch_membership_error(const Rounds& rounds) {
    if (rounds.size() > kMtpBatchMaxSlots) return "mtp batch: invalid membership";
    for (std::size_t i = 0; i < rounds.size(); ++i) {
        if (!rounds[i].drafter) return i == 0 ? "mtp batch: invalid membership" : "mtp batch: invalid round";
        for (std::size_t j = 0; j < i; ++j)
            if (rounds[i].drafter == rounds[j].drafter) return "mtp batch: duplicate drafter";
    }
    return nullptr;
}

struct MtpBatchScratch {
    std::array<int, kMtpBatchMaxSlots> limits{}, counts{};
    std::array<std::array<int32_t, 2>, kMtpBatchMaxSlots> prev{};
    std::array<bool, kMtpBatchMaxSlots> active{};
};

} // namespace strata::core
