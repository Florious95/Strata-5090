#pragma once
#include "strata/kernels/batch_device_table.hpp"
#include "strata/kernels/rope_scaling.hpp"
#if defined(STRATA_GPU_METAL) || defined(STRATA_GPU_SYCL)
#include <stdexcept>
#endif

namespace strata::kernels {
struct BatchGdnShape {
    int channels, key_heads, value_heads;
    int64_t state_stride, conv_offset;
};
#if defined(STRATA_GPU_METAL) || defined(STRATA_GPU_SYCL)
inline bool batch_device_gdn_supported() { return false; }
inline bool batch_device_scores_supported() { return false; }
#define STRATA_UNSUPPORTED_TABLE(name) template <class... A> inline void name(A&&...) { \
    throw std::logic_error("batch row table requires NVIDIA CUDA"); }
STRATA_UNSUPPORTED_TABLE(batch_device_gdn_conv)
STRATA_UNSUPPORTED_TABLE(batch_device_gdn_step)
STRATA_UNSUPPORTED_TABLE(batch_device_gdn_commit)
STRATA_UNSUPPORTED_TABLE(batch_device_ple_copy)
STRATA_UNSUPPORTED_TABLE(batch_device_ple_commit)
STRATA_UNSUPPORTED_TABLE(batch_device_kv_append)
STRATA_UNSUPPORTED_TABLE(batch_device_indexer)
STRATA_UNSUPPORTED_TABLE(batch_device_scores)
STRATA_UNSUPPORTED_TABLE(batch_device_kv_resolve)
STRATA_UNSUPPORTED_TABLE(batch_device_attention)
#undef STRATA_UNSUPPORTED_TABLE
#else
bool batch_device_gdn_supported();
bool batch_device_scores_supported();
void batch_device_gdn_conv(DeviceBatchView v, int begin, int end, int gi, BatchGdnShape g,
                           const float* qkv, const float* weights, float* h, float epsilon, void* stream);
void batch_device_gdn_step(DeviceBatchView v, int begin, int end, int gi, BatchGdnShape g,
                           const float* h, const float* gate, const float* beta, const float* z,
                           const float* gamma, float* y, float epsilon, void* stream);
void batch_device_gdn_commit(DeviceBatchView v, int gi, BatchGdnShape g, const float* qkv,
                             const float* h, const float* gate, const float* beta, void* stream);
// Reuse existing per-row PLE history snapshots as working histories; no new
// sequence/history arena and no change to ple_block's arithmetic.
void batch_device_ple_copy(DeviceBatchView v, int row, float* history, int64_t floats,
                           bool to_slot, void* stream);
void batch_device_ple_commit(DeviceBatchView v, const float* snapshots, int64_t floats, void* stream);
void batch_device_kv_append(DeviceBatchView v, int qi, int begin, int end, const int32_t* steps,
                            const float* k, const float* value, const QsaShapes& s, void* stream);
void batch_device_indexer(DeviceBatchView v, int qi, int begin, int end, const float* raw,
                          const int32_t* steps, const float* gamma, float epsilon, const QsaShapes& s,
                          const RopeScaling& scaling, bool committing, void* stream);
void batch_device_scores(DeviceBatchView v, int qi, int begin, int end, const float* queries,
                         const int32_t* steps, int64_t max_blocks, float* scores,
                         const QsaShapes& s, void* stream);
void batch_device_kv_resolve(DeviceBatchView v, int qi, int begin, int end, const int32_t* ids,
                             const int32_t* steps, int64_t cap, const QsaShapes& s, void* stream);
void batch_device_attention(DeviceBatchView v, int qi, int begin, int end, const float* query,
                            const int32_t* ids, const int32_t* steps, int64_t cap, const QsaShapes& s,
                            float* scratch, float* attn, void* stream);
#endif
} // namespace strata::kernels
