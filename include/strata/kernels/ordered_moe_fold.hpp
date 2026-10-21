#pragma once

#include <cmath>
#include <limits>
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <cuda_runtime.h>
#endif

// A per-output scalar contract, not a claim that CPU/GPU expert producers are
// bit-identical. Inputs must be the same F32 values in the same router-rank
// order. Host witnesses require round-to-nearest and gradual underflow.
namespace strata::kernels::ordered_moe {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_MOE_FOLD_HD __host__ __device__ __forceinline__
#else
#define STRATA_MOE_FOLD_HD inline
#endif

STRATA_MOE_FOLD_HD float add_rn(float a, float b) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    // Explicit PTX deliberately has no .ftz: native_moe.cu is built with
    // --use_fast_math, which must not change this reduction's F32 contract.
    float result;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
#else
    volatile float result = a + b;
    return result;
#endif
}

STRATA_MOE_FOLD_HD float mul_rn(float a, float b) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    float result;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
#else
    volatile float result = a * b;
    return result;
#endif
}

STRATA_MOE_FOLD_HD float fma_rn(float a, float b, float c) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
#elif defined(__HIP_DEVICE_COMPILE__)
    return __builtin_fmaf(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

class Fold {
public:
    // The first multiply rounds separately. Every subsequent expert is one
    // FMA, in router-rank order; shared is one final, unweighted rounded add.
    STRATA_MOE_FOLD_HD Fold(float first_part, float first_weight)
        : sum_(mul_rn(first_part, first_weight)) {}
    STRATA_MOE_FOLD_HD void append(float part, float weight) { sum_ = fma_rn(part, weight, sum_); }
    STRATA_MOE_FOLD_HD float value() const { return sum_; }
    STRATA_MOE_FOLD_HD float with_shared(float shared) const { return add_rn(sum_, shared); }
private:
    float sum_;
};

#undef STRATA_MOE_FOLD_HD
} // namespace strata::kernels::ordered_moe
