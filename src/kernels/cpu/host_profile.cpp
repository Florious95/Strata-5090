#include "strata/kernels/cpu/host_profile.hpp"

namespace strata::kernels::cpu {
namespace { thread_local HostExpertProfile* profile = nullptr; }
HostExpertProfile* host_expert_profile() noexcept { return profile; }
HostExpertProfile* bind_host_expert_profile(HostExpertProfile* p) noexcept {
    auto* previous = profile;
    profile = p;
    return previous;
}
} // namespace strata::kernels::cpu
