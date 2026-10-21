# PR-2: fixed-leaf ordered MoE reduction

## Scope and non-goals

Base: `806939db941f4ddb9bff4961c9dc5cd6e0df7798`.

This PR applies the batch-invariant **reduction** idea to the existing native MoE combine. It does not import DeepInfer kernels, change expert quantization, add a new scheduling mode, move canonical state, or change sampling. CPU Q8_K and GPU Q8_1 expert producers retain their existing arithmetic and throughput paths.

The guarantee is conditional and intentionally narrow:

> Given identical effective F32 expert leaves, router weights, router-rank order and shared contribution, made ready on the ordered stream, the CUDA combine produces the same result independent of row grouping, supported combine variant, or CPU/GPU job completion order.

This is NOT a guarantee that different expert producers produce identical leaves, nor that complete 1->4->1 model generations are bit-identical. The CPU test retains a one-ULP changed-leaf example that can change a greedy argmax. Fixing the final sum cannot erase that difference without changing the expert calculation.

NaN payload bits are not a cross-platform contract. Non-finite results remain subject to the existing model failure checks; no tolerance or sampling gate is relaxed. The host witness requires IEEE F32, round-to-nearest and gradual underflow. CUDA is the qualification target; HIP/Metal/full-model qualification is not claimed by host tests.

## Arithmetic contract

For effective expert leaves `p[i]` and weights `w[i]` in **router rank**, not arrival order:

```text
s[0] = RN32(p[0] * w[0])
s[i] = FMA_RN32(p[i], w[i], s[i-1]), i = 1..k-1
out  = RN32(s[k-1] + shared)             # if shared exists
```

Shared is unweighted by the router. It is added exactly once, after the expert fold. With no shared contribution, the result is `s[k-1]`, not an extra add of zero.

Mixed-hit preparation is also explicit:

```text
effective_part[rank] = RN32(cpu_part[rank] + hit[rank])  # selected GPU-hit ranks
```

The scattered `moe_hit_add` and fused-hit combine now use the same rounded add. All-hit input retains the existing `RN32(+0 + hit)` semantics. Selected destinations must be unique and coherently published before use, as before; this PR does not legalize duplicate destinations or missing stream dependencies.

For the existing gated shared path, the original GPU sigmoid approximation is unchanged. Its scaling multiply rounds separately before the shared add. Equivalence of sigmoid implementations or changed raw gate inputs is outside this fold contract.

## Implementation

- `include/strata/kernels/ordered_moe_fold.hpp`: shared scalar `Fold` and RN add/multiply/FMA primitives.
- `src/kernels/cuda/native_moe.cu`: generic, K10 vec4, fused mixed-hit, all-hit and gated variants use that contract.
- `src/kernels/cuda/s2_expert_grouped.cu`: only the final `moe_hit_add` addition changes to the common primitive. Expert dot products, quantizers, layouts and activations do not change.
- `include/strata/kernels/native_moe.hpp`: public contract and its limits, no API/ABI signature change.

CUDA uses explicit `add.rn.f32`, `mul.rn.f32`, and `fma.rn.f32` PTX **without `.ftz`**. `native_moe.cu` is already built with `--use_fast_math`; implicit contraction/FTZ must not silently change the reduction contract. The old K10 path already documented first-product rounding followed by ordered FMA; generic and gated paths no longer leave that interpretation to compiler contraction.

No allocations, atomically accumulated floating-point sums, additional kernel launches, global synchronizations, routing sorts, or expert replicas are introduced. Existing vec4 loads/stores and rank traversal remain. Ordinary finite results should follow the documented K10 arithmetic; previously compiler-dependent generic/gated or subnormal corner cases can intentionally change bits. Actual throughput and register usage still require CUDA measurement, not an assumption from instruction counts.

## Public CPU witness

`tests/core/ordered_moe_fold_cpu.cpp` directly calls the production helper:

- rank-ordered reference, k=1..15;
- separately rounded first multiply and final shared add;
- an FMA-sensitive cancellation case and reassociation positive control;
- separate gated-scale rounding;
- signed zero and subnormal/halfway boundaries;
- all 1024 K10 CPU/GPU ownership masks with permuted arrivals and 1->4->1 logical row grouping;
- a changed-leaf/greedy-argmax counterexample that must remain different.

The current fixture executes **150617 checks**. Its ownership producers copy the same known leaves; they do not execute expert quantization. A controlled unfused-FMA mutant must fail the FMA-sensitive check. This is test sensitivity, not a claim that a historical CUDA binary was executed and failed.

```bash
c++ -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror -ffp-contract=off \
  -Iinclude tests/core/ordered_moe_fold_cpu.cpp -o /tmp/ordered_moe_fold_cpu
/tmp/ordered_moe_fold_cpu
c++ -std=c++20 -O1 -g -DNDEBUG -Wall -Wextra -Werror -ffp-contract=off \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Iinclude tests/core/ordered_moe_fold_cpu.cpp -o /tmp/ordered_moe_fold_asan
/tmp/ordered_moe_fold_asan
```

CMake target/CTest name: `ordered_moe_fold_cpu` under `STRATA_BUILD_TESTS=ON`. The existing CPU-contract workflow also compiles and executes Release and ASan/UBSan variants.

## Public CUDA witness (separate serialized qualification)

`tests/core/ordered_moe_fold_gpu.cpp` uses production entry points, no model assets:

- widths 7/8/31/2560, k=1/2/10/15;
- single-row production API versus CPU RN bits and captured multi-row variants;
- direct/shared-null/all-hit/scattered/fused/gated-zero cases;
- R1/R4/R8/R12 graph reuse with owner/row permutations, including 6+6 and 3+3+3+3;
- CPU-H2D and GPU-D2D fixed-leaf producers on two nonblocking streams, with actual fork/join ordering;
- dynamic hit destinations/counts, shuffled arrival order, inactive-row and output-tail poison;
- R12 uses the existing scatter fallback when fused-hit support is limited to eight rows;
- subnormal/rounding cases that must survive the production fast-math compilation unit.

It plans **3456 fixtures/replays**. Gated witness uses raw gate zero (exact sigmoid 0.5), so it does not pretend to validate cross-device sigmoid approximations. The complete effective leaf tensor and final shared contribution are identical for each single/multi reference pair.

CMake target/CTest: `ordered_moe_fold_gpu`, NVIDIA CUDA only, exit77 for unavailable device or preservation of the existing 8 GiB test reserve. CPU or host-syntax success is not CUDA execution success.

```bash
cmake -S . -B build-pr2 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build-pr2 --target ordered_moe_fold_gpu
ctest --test-dir build-pr2 -R '^ordered_moe_fold_gpu$' --output-on-failure
```

## Local construction results (controller, not CUDA qualification)

- Standalone Release and ASan/UBSan: 150617 checks each, exit0.
- CMake Release and CMake ASan/UBSan: 8/8 selected CPU regressions each (ordered fold, device layout, Solo, window audit, MTP batching, conversation cache, telemetry, logit bias).
- Unfused-FMA sensitivity mutant: expected exit1 at the FMA-sensitive append check; production helper remains green.
- GPU witness: strict host syntax check with Metal compatibility headers only. No NVIDIA compiler or device execution was performed on this controller.

## Promotion gates and remaining limitations

1. Real NVCC/sm120 compilation and the public GPU witness, with source/binary/toolchain receipts.
2. Existing native multi/fusions parity and canonical C4 device-table GPU tests.
3. Frozen expert residency mapping and fixed numerical profile for any full-model bit comparison; separate that from deliberate CPU/GPU leaf changes under dynamic adaptation.
4. Same-process 1->4->1 lifecycle, unchanged 4x256K / rows12 / Q4 streaming, no invalid windows, exact sampler/commit gates intact.
5. Register usage, per-window timings and three paired throughput rounds; minimum GPU free >=300 MiB. No speedup or no-regression claim before those measurements.

No sampler, expert leaf, GDN, attention, snapshot/rewind or graph-cache implementation is changed here. Host/controller tests are sufficient for the local construction stage only. They cannot certify complete model output determinism or CUDA performance.
