#pragma once

#include <cstdlib>
#include <stdexcept>

namespace strata::kernels {
inline bool parse_batch_deterministic_reduction(const char* value, bool& result) {
    if (!value) { result = false; return true; }
    if ((value[0] == '0' || value[0] == '1') && value[1] == '\0') {
        result = value[0] == '1';
        return true;
    }
    return false;
}

// A process-lifetime numerical profile: initialize before graph construction.
// Changing the environment later cannot change kernels already captured in a
// graph. OFF/ON A/B uses the same binary in separate backend processes.
inline bool batch_deterministic_reduction_enabled() {
    static const bool enabled = [] {
        bool value = false;
        if (!parse_batch_deterministic_reduction(std::getenv("STRATA_BATCH_DETERMINISTIC_REDUCTION"), value))
            throw std::invalid_argument("STRATA_BATCH_DETERMINISTIC_REDUCTION must be 0 or 1");
        return value;
    }();
    return enabled;
}
} // namespace strata::kernels
