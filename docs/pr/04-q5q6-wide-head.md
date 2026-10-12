# PR-4 · Q5_K / Q6_K wide LM Head

- Branch: `pr/04-q5q6-wide-head`.
- Execution base: accepted PR-2b + PR-6 integration SHA; currently pending. Native primitive development can be isolated before caller integration.
- Goal: project 9–12 columns with one logical large-weight pass while retaining small Q8 activation quantization, 4-warp reduction and per-column arithmetic order.
- Scope: `include/strata/kernels/native_mmvq.hpp`, `src/kernels/cuda/native_mmvq.cu`, `src/kernels/mmvq_k_wide_parity.cpp`, head dispatch in `verify.cpp` and required batch fields/caller, scoped CMake registration.
- Required observability: actual `head_type` and `head_wide_active`; actual frozen model head is type13 Q5_K, [2560,248320],437043200 bytes. Support type14 Q6_K without relying on it being present in the real model.
- Non-goals: replacing/repacking GGUF, changing output math or reduction order, quantizing the head differently, globally enabling unsafe wider verifier rows, removing NaN guard.
- Contracts: bitwise GPU baseline versus wide outputs for each T=9/10/11/12, direct and captured graphs, full real dimensions, odd/tail geometries and dual-stream isolation. Compare each column; CPU numerical tolerance is not a bitwise GPU contract.
- Historical evidence:196.33 versus191.38 pooled (+2.58%); pairs +10.067/+2.748/−4.335%. Retain third-ON682ms verify spike. Avoiding a second logical437MB payload pass is not a measured GDDR byte counter. Superseded Q6-only candidate was a no-op on this Q5 asset.
- New acceptance: fresh GPU receipts plus C1 and three same-binary C2 ON/OFF pairs; activation requires native dispatch evidence, not flag presence.
- Rollback: disable wide projection to split ≤8-row fallback; capacity-safe nonfinite guards stay enabled.

Freeze actual base/head SHA and independent GPU window; record test geometries/counts/exits and binary hashes. See [workflow](../FORK-WORKFLOW.md).
