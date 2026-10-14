// CPU-only accounting contract: nested PLE/stage bins are never double-counted,
// missing/negative residuals remain observable. No model/CUDA device is required.
#include "strata/core/verify.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <stdexcept>

using strata::core::BatchHostProfile;
void require(bool ok) { if (!ok) throw std::runtime_error("host accounting mismatch"); }

int main() {
    try {
        BatchHostProfile p;
        require(p.run_other_ns() == 0 && p.stage_other_ns() == 0 && !p.graph_cache_hit);
        std::mt19937 rng(5090);
        for (int i = 0; i < 10000; ++i) {
            auto ns = [&] { return int64_t(rng() % 1000000000u); };
            p.evict_ns = ns(); p.capture_exec_ns = ns(); p.capture_commit_ns = ns();
            p.ple_gather_ns = ns(); p.stage_data_ns = p.ple_gather_ns + ns();
            const int64_t stage_gap = ns(), run_gap = ns();
            p.stage_ns = p.evict_ns + p.capture_exec_ns + p.capture_commit_ns + p.stage_data_ns + stage_gap;
            p.context_enter_ns = ns(); p.context_exit_ns = ns();
            p.launch_ns = ns(); p.query_ns = ns(); p.ring_ns = ns(); p.tail_sync_ns = ns();
            p.copy_sync_ns = ns(); p.collect_ns = ns(); p.sample_ns = ns(); p.out_ns = ns(); p.progress_ns = ns();
            const int64_t known = p.context_enter_ns + p.context_exit_ns + p.stage_ns + p.launch_ns + p.query_ns +
                p.ring_ns + p.tail_sync_ns + p.copy_sync_ns + p.collect_ns + p.sample_ns + p.out_ns + p.progress_ns;
            p.run_ns = known + run_gap;
            require(p.stage_other_ns() == stage_gap && p.run_other_ns() == run_gap);
            p.run_ns = known - 1;
            require(p.run_other_ns() == -1); // fail-closed consumer must see this, not clamp it
        }
        p = {};
        require(p.run_ns == 0 && p.ple_gather_ns == 0 && p.run_other_ns() == 0 && p.stage_other_ns() == 0);
        using Clock = std::chrono::steady_clock;
        static_assert(Clock::is_steady);
        int64_t minimum = INT64_MAX;
        auto previous = Clock::now();
        for (int i = 0; i < 10000; ++i) {
            const auto now = Clock::now();
            const auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(now-previous).count();
            require(d >= 0);
            if (d > 0 && d < minimum) minimum = d;
            previous = now;
        }
        std::printf("PASS: 10000 host accounting cases; steady clock observed min positive delta=%lld ns (unit, not accuracy)\n",
                    (long long) minimum);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
