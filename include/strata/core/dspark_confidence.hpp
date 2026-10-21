#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace strata::core {

struct DsparkDepthPlan {
    int rows = 0, observations = 0;
    uint64_t generation = 0, step = 0, history_through_window = 0;
    double observed_rate = 0.0;
};

// Historical acceptance proxy, not DeepSeek's trained Confidence Head.
// Each owner/request has its own clock. For target step N, only outcomes
// through N-2 enter the 16-window estimate; N-1 remains quarantined. Planning
// takes neither current draft tokens/probabilities nor current target output.
class DsparkConfidence {
public:
    static constexpr int max_rows = 8, history_windows = 16, cold_rows = 3;

    bool reset(uint64_t generation) {
        if (!generation) return false;
        generation_ = generation;
        completed_ = last_window_ = 0;
        history_ = {};
        return true;
    }

    bool observe(uint64_t generation, uint64_t window, int useful, int kept) {
        if (!generation_ || generation != generation_ || !window || window <= last_window_ ||
            useful < 1 || useful > max_rows || kept < 1 || kept > useful ||
            completed_ == std::numeric_limits<uint64_t>::max()) return false;
        const uint64_t step = completed_ + 1;
        history_[step % history_.size()] = {step, window, useful, kept};
        completed_ = step;
        last_window_ = window;
        return true;
    }

    DsparkDepthPlan plan(int maximum) const {
        DsparkDepthPlan out;
        if (!generation_ || maximum < 1 || maximum > max_rows ||
            completed_ == std::numeric_limits<uint64_t>::max()) return out;
        out.generation = generation_;
        out.step = completed_ + 1;
        out.rows = std::min(cold_rows, maximum);
        if (completed_ < 2) return out;
        const uint64_t last = completed_ - 1; // N-2, never the just-completed N-1
        const uint64_t first = last > history_windows ? last - history_windows + 1 : 1;
        std::array<int, max_rows> trials{}, accepted{};
        int all_trials = 0, all_accepted = 0;
        for (uint64_t step = first; step <= last; ++step) {
            const auto& item = history_[step % history_.size()];
            if (item.step != step) return {}; // stale/missing history is not confidence
            ++out.observations;
            out.history_through_window = item.window;
            // Proposal d is a conditional trial only if all earlier proposals
            // matched. Rejected suffixes and unverified rows are censored.
            for (int d = 1; d < item.useful && d <= item.kept; ++d) {
                ++trials[d]; ++all_trials;
                if (d < item.kept) { ++accepted[d]; ++all_accepted; }
            }
        }
        if (all_trials) out.observed_rate = double(all_accepted) / all_trials;
        if (out.observations < 4 || !all_trials) return out;
        // Beta(1,1) smoothing; unseen deeper positions use the pooled past
        // estimate so a short cold window does not permanently block growth.
        const double fallback = double(all_accepted + 1) / (all_trials + 2);
        double survival = 1.0;
        out.rows = std::min(2, maximum);
        for (int d = 1; d < maximum; ++d) {
            survival *= trials[d] ? double(accepted[d] + 1) / (trials[d] + 2) : fallback;
            if (survival < 0.5) break;
            out.rows = std::max(out.rows, d + 1);
        }
        return out;
    }

private:
    struct Observation { uint64_t step = 0, window = 0; int useful = 0, kept = 0; };
    std::array<Observation, history_windows + 1> history_{};
    uint64_t generation_ = 0, completed_ = 0, last_window_ = 0;
};

} // namespace strata::core
