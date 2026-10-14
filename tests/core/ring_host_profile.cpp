#include "strata/kernels/cpu/host_profile.hpp"
#include <atomic>
#include <cstdio>
#include <thread>

namespace c = strata::kernels::cpu;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line%d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
    CHECK(c::host_expert_profile() == nullptr);
    c::HostExpertProfile a, b;
    {
        c::HostExpertProfileScope outer(&a);
        CHECK(c::host_expert_profile() == &a);
        { c::HostExpertProfileScope disabled(nullptr); CHECK(c::host_expert_profile() == &a); }
        { c::HostExpertProfileScope inner(&b); CHECK(c::host_expert_profile() == &b); }
        CHECK(c::host_expert_profile() == &a);
        std::atomic<bool> isolated{false};
        std::thread t([&] {
            const bool empty = c::host_expert_profile() == nullptr;
            c::HostExpertProfile own;
            { c::HostExpertProfileScope scope(&own); isolated.store(empty && c::host_expert_profile() == &own); }
            isolated.store(isolated.load() && c::host_expert_profile() == nullptr);
        });
        t.join(); CHECK(isolated.load());
        CHECK(c::host_expert_profile() == &a);
    }
    CHECK(c::host_expert_profile() == nullptr);
    for (int trial = 0; trial < 10000; ++trial) {
        c::RingHostProfile r;
        const int steps = 1 + trial % c::RingHostProfile::kMaxSteps;
        for (int i = 0; i < steps; ++i) {
            c::RingStepProfile s;
            s.layer = i / 2; s.group = i % 2;
            s.wait_ns = int64_t(trial + 1) * 1000000001LL;
            s.callback_ns = 50 + i; s.publish_ns = 2 + i;
            r.add(s);
        }
        r.total_ns = r.wait_ns + r.callback_ns + r.publish_ns + 123;
        CHECK(r.steps == steps && r.other_ns() == 123);
        int64_t wait = 0, callback = 0, publish = 0;
        for (int i = 0; i < r.steps; ++i) {
            wait += r.samples[i].wait_ns; callback += r.samples[i].callback_ns; publish += r.samples[i].publish_ns;
        }
        CHECK(wait == r.wait_ns && callback == r.callback_ns && publish == r.publish_ns);
        auto& d = r.expert.dispatch;
        d.plan_ns = 4; d.actq_ns = 5; d.jobs_ns = 6; d.run_ns = 7; d.total_ns = 31;
        CHECK(d.other_ns() == 9);
        auto& p = r.expert.phases;
        p.park_before_ns = 1; p.publish_ns = 2; p.host_work_ns = 3; p.wait_done_ns = 4; p.park_after_ns = 5;
        p.total_ns = 21; p.native_phase_ns = 21; p.ff_quant_ns = 7; p.native_ns = 32;
        CHECK(p.other_ns() == 6 && p.native_other_ns() == 4);
        --r.total_ns; CHECK(r.other_ns() == 122); // exact64bit arithmetic, no float rounding
        r.total_ns = 0; CHECK(r.other_ns() < 0);  // never clamp a broken receipt
        d.total_ns = 0; CHECK(d.other_ns() < 0);
        p.total_ns = 0; p.native_ns = 0;
        CHECK(p.other_ns() < 0 && p.native_other_ns() < 0);
        r = {}; CHECK(r.steps == 0 && r.total_ns == 0 && r.expert.dispatch.calls == 0);
    }
    std::puts("PASS: 10000 ring/dispatch/native-phase integer closures; TLS disabled/nested/thread isolation");
    return 0;
}
