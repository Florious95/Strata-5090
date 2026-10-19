# v0.2.0-beta + C2 golden port

## Source and scope

This branch merges official release source `3d1fb48090e8f2cefc22d7b5845fb7f5b71f33fa`
(tag `v0.2.0-beta`, engine version `0.1.45.3`) with the qualified C2 source
`265e1aeb6fd36e1a675a4476b9c1670423f77bf9`. Their common base is
`296b59d2f73c8f04a805df29c2fa97655adace78`.

It preserves IQ3_XXS/Q4 KV operation, adaptive spec6/rows12, private slot target and
MTP state, direct admission, narrow dense parallel streams, wide Q5_K/Q6_K head,
QSA same-slot tiles, image SHA references/leases and Token LCP slot affinity.
The upstream planner, SM120 defaults, native IL kernels, lifecycle/queue controls,
Hadamard formats and layer-pipeline scheduling remain present.

**This is a source port, not a measured GPU upgrade or production deployment.**
No performance percentage is claimed. The retired production remains v0.1.43;
PR9 was an experiment, not a production pin replacement.

## Resolution ledger

The real merge had six conflicted files / 38 conflict blocks:

| File | Blocks | Resolution |
|---|---:|---|
| CMakeLists.txt | 2 | Keep host profiling and upstream Hadamard sources; retain both upstream and gold parity targets. |
| include/strata/core/verify.hpp | 1 | Preserve gold batch fields and 16-row output storage, plus upstream atomic scheduling state. |
| serve/server.py | 6 | Combine stage/batch telemetry and token gaps; retain queue deadlines, cancellation, reader args, image leases and LCP routing. |
| src/core/verify.cpp | 14 | Preserve official side mixer/QIL/QSA row paths and gold segmentation/private scratch; move routed Hadamard quantization into the shared-pre helper. |
| src/kernels/native_expert_parity.cpp | 3 | Retain synthetic, Hadamard and 16-row-wide modes. |
| src/program/generate.cpp | 12 | Keep pipeline grouping/planned prefill and direct slots/private drafters, adaptive rows, RAM park-before-overwrite and trace accounting. |

Additional semantic fixes (not just marker deletion):

- Native Q8/IL shape APIs accept at most eight columns. Allocate plain images as
  one-row bytes times verifier capacity; size each private IL image for 4..8
  columns, never call its allocator with 12/16 columns. Both dense groups retain
  distinct quantized images. The CPU scratch contract exercises all supported
  partitions through 16 rows.
- Preserve each slot's M-RoPE scope in upstream's factored index-append helper.
- Shared experts/router read unrotated rows; routed APR experts alone consume the
  rotated image, including when shared work is deferred until after ring publish.
- Direct slots keep whole-snapshot conversation transfer. Upstream prefix
  borrowing is disabled there because its donor guard belongs to the main-session
  park budget, not the private-slot park budget.
- Telemetry accepts gold's 12/16 work rows; these are not emitted tokens. Request
  summaries keep upstream first-chunk accounting and gold exact native token-gap
  aggregation together.
- The upstream portable lifeline watcher included changing process status in its
  identity token. Excluding R/S/status flags while rejecting zombies restores
  kill-parent cleanup on macOS. Physical child-process and identity tests cover it.
- The release's no-site CLI smoke expected a missing-profile event although the
  new dependency preflight runs first. It now verifies that actual actionable
  exit5 refusal. Runtime smoke accounting accepts platform skips but counts only
  executed passes toward its minimum; regression tests reject skip inflation and
  nonzero exits. These changes do not change serving behavior.
- Vision argument tests stub the spawn boundary instead of starting a real
  watcher around a fake process without a PID. The lease-failure test explicitly
  selects `fit_max_tokens=False`, preserving upstream's new default cap fitting.

## Deliberate limits

- Gold fast admission still requires **2..4 slots, single NVIDIA GPU, full-resident
  int8/Q4 KV**. Direct admission remains **C2/Q4 only**. No guard was removed to
  claim C8, streaming+12 rows, or grow+batch support.
- Every requested capability comparison must keep logical `--max-context 262144`.
  Reducing it to 32K is not KV streaming. Unsupported combinations are reference
  arms, not gold successes.
- The new planner plans expert-file reads and H2D/ring/cache loans, not KV. The
  default gold arena expert source bypasses its file-source guard.
- Native IL kernels still have no 12-column implementation; segmented fallback
  and the separate gold wide head remain necessary.
- Fused prefill is not bitwise-equivalent. First qualify exact decode with
  `STRATA_PF_FUSED=0`, then test fused prefill quality/TTFT separately. Check the
  exact environment spelling in the release source before constructing a recipe.
- SM120's side mixer has `!batch_rec_`; single-flow PRO5000 measurements do not
  establish benefits for partly-resident IQ3_XXS C2 on an RTX5090.
- No NVFP4 weights/KV migration, driver/toolkit change, model download, service
  restore, or GPU-owner preemption is included.

## Controller validation and GPU handoff

Use pinned requirements and Python 3.12. Full server discovery, helper contracts
(Release + ASan/UBSan), portable host CTest targets, analyzer/tokenizer contracts,
and mocked Linux NVFP4 setup tests are recorded separately in the delivery receipt.
Controller results: server **944 tests / 929 pass / 15 skip / exit0**; six helper
programs each compile/run in Release and ASan/UBSan; portable CTest **17/17**;
selected analyzer/tokenizer contracts **45 / 42 pass / 3 skip** plus four
skip-accounting regressions; Linux-platform
mocked NVFP4 setup **21/21**. The 43-test red-to-green subset also passes (one
platform skip). Skipped tokenizer/SDK tests are not passes. The initial red log
(939 tests, four failures/one error) is retained with the corrections above.

The root CPU-only CMake all-target build is not portable to the controller's ARM
host (x86 intrinsic sources); an experimental x86/native-experts-off build also
exposes missing upstream AVX1 link objects. Neither is reported as a successful
native engine build. Host syntax checks using Metal compatibility declarations are
only syntax checks, **not CUDA compilation/linking or GPU parity**.

Before any model benchmark, an authorized bench owner must build this exact head
with CUDA13.3 / sm120 (NVFP4 W4A4 TU alone sm120a), record toolchain/source/binary
hashes and pass private GDN, M-RoPE, QSA causal/poisoned replay, NaN and slot-isolation
checks. Then run frozen-ID C2 258K/queued4 paired A/B, trace-off performance and
trace-on diagnostics separately. Match cache bytes/membership, flags, CPU workers,
clocks, images/vision reservation, fixture and renderer IDs. A port-vs-stock test
and a pure-version upgrade test are different experiments.

GPU resources remain owned by the video team until an explicit handoff window.
An unused HTTP port is not GPU isolation.
