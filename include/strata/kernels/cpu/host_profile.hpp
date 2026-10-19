#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

// Host-only, optional per-window sink. Nested bins are never added to ring wall time.
struct PoolPhaseProfile {
    int64_t calls = 0, total_ns = 0;
    int64_t park_before_ns = 0, publish_ns = 0, host_work_ns = 0, wait_done_ns = 0, park_after_ns = 0;
    int64_t native_calls = 0, native_ns = 0, native_phase_ns = 0, ff_quant_ns = 0;
    int64_t native_other_ns() const { return native_ns - native_phase_ns - ff_quant_ns; }
    int64_t other_ns() const {
        return total_ns - park_before_ns - publish_ns - host_work_ns - wait_done_ns - park_after_ns;
    }
};

struct DispatchProfile {
    int64_t calls = 0, total_ns = 0;
    int64_t plan_ns = 0, actq_ns = 0, jobs_ns = 0, run_ns = 0;
    int64_t zero_cpu_calls = 0, cpu_jobs = 0, cpu_needed_tokens = 0, quantized_tokens = 0;
    int64_t other_ns() const { return total_ns - plan_ns - actq_ns - jobs_ns - run_ns; }
};

struct HostExpertProfile {
    PoolPhaseProfile phases;
    DispatchProfile dispatch;
};

struct RingStepProfile {
    int layer = -1, group = -1;
    int64_t wait_ns = 0, callback_ns = 0, publish_ns = 0;
    int64_t plan_ns = 0, actq_ns = 0, jobs_ns = 0, run_ns = 0;
};

struct RingHostProfile {
    static constexpr int kMaxSteps = 128;
    int steps = 0;
    int64_t total_ns = 0, wait_ns = 0, callback_ns = 0, publish_ns = 0;
    HostExpertProfile expert;
    std::array<RingStepProfile, kMaxSteps> samples{};
    int64_t other_ns() const { return total_ns - wait_ns - callback_ns - publish_ns; }
    void add(const RingStepProfile& sample) {
        samples[(std::size_t) steps++] = sample;
        wait_ns += sample.wait_ns;
        callback_ns += sample.callback_ns;
        publish_ns += sample.publish_ns;
    }
};

HostExpertProfile* host_expert_profile() noexcept;
HostExpertProfile* bind_host_expert_profile(HostExpertProfile* profile) noexcept;

class HostExpertProfileScope {
public:
    explicit HostExpertProfileScope(HostExpertProfile* p) : bound_(p != nullptr) {
        if (bound_) previous_ = bind_host_expert_profile(p);
    }
    ~HostExpertProfileScope() { if (bound_) bind_host_expert_profile(previous_); }
    HostExpertProfileScope(const HostExpertProfileScope&) = delete;
    HostExpertProfileScope& operator=(const HostExpertProfileScope&) = delete;
private:
    bool bound_;
    HostExpertProfile* previous_ = nullptr;
};

} // namespace strata::kernels::cpu
