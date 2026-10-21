# PR-4: exact compact router Top10 (experimental, default off)

Frozen base: `806939db941f4ddb9bff4961c9dc5cd6e0df7798`.
Branch: `feat/deepseek-pr4-deepselect-topk`.

## Contract and non-goals

The existing Flash-Next path already computes the 512-expert router, selects
exactly ten experts, and publishes only their IDs/weights before expert GEMMs.
It does **not** run all 512 experts. Dropping any genuinely selected expert to
avoid a cache miss changes the model and is outside this PR.

This experiment reduces candidate-selection work, not the selected expert set
or required PCIe bytes. No PCIe-latency or end-to-end speed improvement is claimed.
It borrows DeepSelect's conservative filter/compact principle, not its CUDA
implementation, precision changes, or SM100-only kernels. It adds no dependency.

## Exact conservative bound

1. Keep the existing full FP32 softmax and its warp reduction order.
2. Use the upper 16 **integer bits** of each original FP32 probability only as
   a monotonic bucket key; never round the probabilities or recompute weights.
3. Pair lanes `j` and `j+16`. The minimum of these 16 disjoint groups' maximum
   keys is a lower bound: at least 16 values survive, so a lower bucket cannot
   contain any of the top ten. Keep every value in the boundary bucket.
4. If 16..128 candidates survive, compact their original 16-bit IDs and FP32
   values into a bounded table (6 KiB shared memory per eight-row block).
   Each lane then searches at most four candidates, instead of sixteen.
5. Uniform/dense buckets and invalid rows take the full original selection
   body. In particular, >128 candidates are **never truncated**.
6. Rank by exact FP32 probability, ties by original expert ID. Accumulate the
   selected denominator in the **original winning expert lane**, not the compact
   lane or output rank: otherwise FP32 weights can change even with identical IDs.

The old single/multi/256x8 kernels remain intact and remain the default. Only
512x10 NVIDIA calls can select the new kernel. No slot, verifier, KV, sampler,
state/graph budget, checkpoint, or expert-placement policy is changed.

## FTZ and graph safety

Control operations are integer bit casts, comparisons, counts, prefix sums and
shuffles. There is **no** subnormal floating-point integer-addition trick and no
new `--ftz` or global fast-math option. Rows containing negative values, nonfinite
values or positive subnormal probabilities fall back, since FTZ float ordering
may differ from integer ordering. If an existing FTZ softmax already emitted
zero, that zero may be safely filtered according to the normal bound.

The kernel has fixed grid geometry and bounded shared memory; no allocation,
host synchronization, default-stream work, or JIT occurs during capture.
Configure `STRATA_ROUTER_COMPACT_TOPK=0|1` before engine startup. It defaults to
zero. Invalid flag values, a disabled native router, non-NVIDIA backends or a model
other than 512 experts/top10 are rejected. READY reports `router_compact_topk`.
Existing graphs retain their chosen kernel; live reconfiguration is unsupported.

## Public tests

- `router_compact_cpu`: 1,346,831 checks / 6,011 probability rows, both branches;
  exact ranked IDs, original-lane weight-bit model, boundary ties, dense fallback,
  16/128/129 boundaries, strict configuration, invalid/subnormal guards. Negative
  controls show that prefix-only TopK and rank-lane normalization are incorrect.
  These are CPU control/selection tests, **not** CUDA normalization proof.
- `router_compact_gpu`: asset-free real CUDA test, 360 planned graph fixtures,
  nine row counts including 1/4/12 and partial-block17, dynamic replay inputs,
  actual production dispatch, candidate-count/branch diagnostics, bitwise IDs
  and weights against the original kernels, original-lane multi-winner case,
  inactive-row/guard poison and argument refusal. Exit77 without CUDA.

CPU CMake setup (no GPU/model/downloads):

```sh
cmake -S . -B build-router-cpu -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-router-cpu --target router_compact_cpu slot_solo_cpu \
  batch_device_layout_cpu conversation_cache_test
ctest --test-dir build-router-cpu --output-on-failure \
  -R '^(router_compact_cpu|slot_solo_cpu|batch_device_layout_cpu|conversation_cache_test)$'
```

Use `-fsanitize=address,undefined -fno-omit-frame-pointer` for the sanitizer build;
keep `-DNDEBUG` and the test's explicit checks. The standalone fast-math build
also checks integer controls with `-ffast-math`, but is not an NVCC FTZ test.

## Serial qualification, not performed on the Mac controller

1. Compile the frozen PR revision on CUDA/sm120, record toolchain/binary/source
   hashes and ptxas registers/shared memory. Run `router_compact_gpu` plus existing
   `router_top10_parity`, `native_multi_parity`, and `route_window_parity`.
2. Benchmark compact and fallback distributions separately for rows1/4/12.
   The extra bound/compaction work can be slower than the existing tiny router;
   do not enable by default without a measured benefit.
3. Same-model/same-placement full logits/output IDs and exact sampling parity;
   no fewer routed expert occurrences or changed miss-list semantics are allowed.
4. Same-PID `1->4->1` lifecycle with Solo+row table, fixed configuration, no
   restarts and minimum free GPU memory >=300MiB. Separate cold/hot graph timing.
5. Preserve default-off status until independent serial GPU acceptance; no
   production deployment, merge or performance claim is authorized by CPU tests.

Local scope evidence is stored outside the source worktree in the controlled
PR runtime directory. CUDA build/runtime and full-model performance remain
unjudgeable on the Mac. The external architecture-wiki tool checkout prescribed
by the workflow is absent on this controller; no zero-finding scanner claim is
made. Manual closure is router API -> layer/verify callers -> startup dispatch,
with CPU geometry/cache regressions and the explicit GPU gate above.
