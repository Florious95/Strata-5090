#pragma once

// Non-owning C4 ABI: canonical state stays in its original slot. Static tables
// and pinned window/commit staging have fixed addresses for the verifier lifetime.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_stream.hpp"
#include <cstddef>
#include <cstdint>
#include <type_traits>
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <cuda_runtime.h>
#endif

namespace strata::kernels {
inline constexpr int kDeviceBatchSlots = 4, kDeviceBatchRows = 12, kDeviceBatchQsaLayers = 12;

struct alignas(16) DeviceTargetSlot {
    float* gdn_base = nullptr;
    float* ple_hist = nullptr;
    int64_t max_cells = 0;
    uint64_t address_epoch = 0;
    int32_t gdn_ord0 = 0, qsa_ord0 = 0;
};
// Writable pool capability borrowed directly from canonical QsaState, not from
// QsaAttnPools (an attention-only const reader). Const descriptors freeze pointer
// identities, not the storage that append/resolve must update. Keep the same
// nine-pointer layout as the reader without permitting reader -> writer casts.
struct DeviceQsaPools {
    uint16_t *k_pool = nullptr, *v_pool = nullptr;
    int8_t *k_q = nullptr, *v_q = nullptr;
    uint16_t *k_scale = nullptr, *v_scale = nullptr;
    uint8_t *k_q4 = nullptr, *v_q4 = nullptr;
    const int32_t* page_table = nullptr;
#if defined(__CUDACC__) || defined(__HIPCC__)
    __host__ __device__
#endif
    QsaAttnPools read_only() const {
        return {k_pool, v_pool, k_q, v_q, k_scale, v_scale, k_q4, v_q4, page_table};
    }
};
static_assert(sizeof(DeviceQsaPools) == sizeof(QsaAttnPools));
static_assert(offsetof(DeviceQsaPools, k_q4) == offsetof(QsaAttnPools, k_q4));
static_assert(std::is_standard_layout_v<DeviceQsaPools> && std::is_trivially_copyable_v<DeviceQsaPools>);
struct alignas(16) DeviceQsaSlot {
    DeviceQsaPools pools;
    KvHostPools host;
    KvStreamMap map;
    float *idx_tail = nullptr, *idx_dead = nullptr, *idx_pooled = nullptr;
    int32_t* idx_block_pos = nullptr;
    const float *cos_tab = nullptr, *sin_tab = nullptr;
    float* tail_snapshot = nullptr;
    int64_t max_cells = 0;
    uint32_t flags = 0, reserved = 0;
    uint64_t address_epoch = 0;
};
inline constexpr uint32_t kDeviceQsaQ4 = 1, kDeviceQsaRotated = 2, kDeviceQsaStreamed = 4;
struct alignas(16) DeviceSlotTable {
    DeviceTargetSlot target[kDeviceBatchSlots];
    DeviceQsaSlot qsa[kDeviceBatchQsaLayers][kDeviceBatchSlots];
};
struct WindowRow { int32_t slot = -1, segment = -1, local_row = -1, reserved = 0; };
struct alignas(16) WindowSegment {
    int32_t slot = -1, first = 0, length = 0, reserved = 0;
    uint64_t request_generation = 0, owner_address_epoch = 0;
};
struct alignas(16) DeviceWindowLayout {
    uint32_t rows = 0, segments = 0, dense_cut = 0, profile = 0;
    uint64_t sequence = 0, table_epoch = 0;
    WindowRow row[kDeviceBatchRows];
    WindowSegment segment[kDeviceBatchSlots];
};
struct alignas(16) DeviceCommitLayout {
    uint64_t sequence = 0, table_epoch = 0;
    int32_t keep_by_segment[kDeviceBatchSlots] = {};
};
struct DeviceBatchView {
    const DeviceSlotTable* table = nullptr;
    const DeviceWindowLayout* window = nullptr;
    const DeviceCommitLayout* commit = nullptr;
    const uint64_t* table_epoch = nullptr;
    uint32_t* status = nullptr; // separate mapped word; nonzero makes the window fatal
};

enum DeviceBatchError : uint32_t {
    kBatchBadDescriptor = 1, kBatchBadEpoch = 2, kBatchBadOwner = 4,
    kBatchBadRange = 8, kBatchBadCommit = 16, kBatchBadQsa = 32
};

static_assert(sizeof(void*) == 8, "device table ABI requires 64-bit pointers");
static_assert(sizeof(DeviceTargetSlot) == 48 && sizeof(DeviceQsaSlot) == 288);
static_assert(sizeof(WindowRow) == 16 && sizeof(WindowSegment) == 32);
static_assert(sizeof(DeviceSlotTable) == 14016 && sizeof(DeviceWindowLayout) == 352 && sizeof(DeviceCommitLayout) == 32);
static_assert(sizeof(DeviceSlotTable) + sizeof(DeviceWindowLayout) + sizeof(DeviceCommitLayout) == 14400);
static_assert(offsetof(WindowSegment, owner_address_epoch) == 24);
static_assert(offsetof(DeviceWindowLayout, row) == 32 && offsetof(DeviceWindowLayout, segment) == 224);
static_assert(std::is_standard_layout_v<DeviceSlotTable> && std::is_trivially_copyable_v<DeviceSlotTable>);
static_assert(std::is_standard_layout_v<DeviceWindowLayout> && std::is_trivially_copyable_v<DeviceWindowLayout>);
static_assert(std::is_standard_layout_v<DeviceCommitLayout> && std::is_trivially_copyable_v<DeviceCommitLayout>);

#if defined(__CUDACC__) || defined(__HIPCC__)
__device__ inline void batch_device_fail(const DeviceBatchView& v, uint32_t error) {
    if (v.status) atomicOr(v.status, error);
}
// Every decision is CTA-uniform. Inactive segments are no-ops, NOT invalid owners.
__device__ inline const WindowSegment* batch_device_segment(const DeviceBatchView& v, int index,
                                                           int begin, int end, bool committing = false) {
    if (!v.table || !v.window || !v.table_epoch || !v.status || v.window->segments < 1 ||
        v.window->segments > kDeviceBatchSlots ||
        v.window->rows < 1 || v.window->rows > kDeviceBatchRows || index < 0 || index >= kDeviceBatchSlots) {
        batch_device_fail(v, kBatchBadDescriptor); return nullptr;
    }
    if (index >= (int) v.window->segments) return nullptr;
    if (!v.window->sequence || !v.window->table_epoch || v.window->table_epoch != *v.table_epoch) {
        batch_device_fail(v, kBatchBadEpoch); return nullptr;
    }
    const auto& s = v.window->segment[index];
    if (s.slot < 0 || s.slot >= kDeviceBatchSlots) { batch_device_fail(v, kBatchBadOwner); return nullptr; }
    if (!s.owner_address_epoch || s.owner_address_epoch != v.table->target[s.slot].address_epoch) {
        batch_device_fail(v, kBatchBadEpoch); return nullptr;
    }
    if (s.length < 1 || s.length > 8 || s.first < 0 || s.first > (int)v.window->rows ||
        s.length > (int)v.window->rows - s.first || !s.request_generation) {
        batch_device_fail(v, kBatchBadRange); return nullptr;
    }
    if (s.first >= end || s.first + s.length <= begin) return nullptr;
    if (s.first < begin || s.first + s.length > end) { batch_device_fail(v, kBatchBadRange); return nullptr; }
    if (committing && (!v.commit || v.commit->sequence != v.window->sequence ||
        v.commit->table_epoch != v.window->table_epoch || v.commit->keep_by_segment[index] < 1 ||
        v.commit->keep_by_segment[index] > s.length)) {
        batch_device_fail(v, kBatchBadCommit); return nullptr;
    }
    return &s;
}
__device__ inline const WindowSegment* batch_device_row(const DeviceBatchView& v, int row, int begin, int end) {
    if (!v.window || row < 0 || row >= kDeviceBatchRows || row >= (int)v.window->rows ||
        row < begin || row >= end) {
        batch_device_fail(v, kBatchBadDescriptor); return nullptr;
    }
    const auto& r = v.window->row[row];
    const auto* s = batch_device_segment(v, r.segment, begin, end);
    if (!s || r.slot != s->slot || r.local_row < 0 || r.local_row >= s->length ||
        row != s->first + r.local_row) {
        batch_device_fail(v, kBatchBadDescriptor); return nullptr;
    }
    return s;
}
__device__ inline const DeviceQsaSlot* batch_device_qsa(const DeviceBatchView& v, const WindowSegment& s, int qi) {
    if (qi < 0 || qi >= kDeviceBatchQsaLayers) { batch_device_fail(v, kBatchBadQsa); return nullptr; }
    const auto& q = v.table->qsa[qi][s.slot];
    if (q.address_epoch != s.owner_address_epoch ||
        q.flags != (kDeviceQsaQ4 | kDeviceQsaRotated | kDeviceQsaStreamed)) {
        batch_device_fail(v, kBatchBadQsa); return nullptr;
    }
    return &q;
}
#endif
} // namespace strata::kernels
