# PR-2b · Adaptive12 and Early-Ring bundle

- Branch: `pr/02b-adaptive12-early-ring`.
- Execution base: accepted PR-1 + PR-6 integration SHA; currently pending.
- Goal: per-slot adaptive spec6 and merged rows≤12; dispatch CPU experts early and preserve wide-window dense/ring overlap.
- Scope: `generate.cpp`, `verify.hpp/.cpp`, batch segment/layout state, CPU expert row-capacity templates and tests, required native routing/accounting fields. New caller/interface capacity changes must compile as one unit.
- Required behavior: correct token→slot state and per-slot accepted prefixes, valid >8-row merged graph, full-window expert plan, safe partial commit and private residual state, guards active for every row.
- Non-goals: widening legacy primitive `kVerifyMaxT=8` indiscriminately; Narrow Dense, Q5/Q6 wide-head, QSA Tile, optional profiling ABI or PLE keyed prefetch.
- Contracts: 9/10/11/12 row capacity, segment/acceptance boundaries, CPU max-row capacity, GDN/state parity and poisoning; routed=VRAM+CPU+offload occurrence balance and native rows/output accounting. Preserve true C1 unchanged.
- Historical evidence: pooled177.96 versus contemporaneous172.89 (+2.93%) was a spec6/rows12 versus spec4/rows8 **bundle**, not Early-Ring-only attribution; paired gains +12.07/+0.503/−2.862% remain retained.
- New acceptance: scoped CPU/service/CUDA/GPU tests plus three fresh same-binary bundle ON/OFF pairs. Freeze actual CLI draft depth, row width, reserve and assets, not just INFO labels.
- Rollback: disable/revert bundle with a ≤8-row compatible fallback; do not disable the nonfinite guard.

Fill accepted base SHA, commands/counts and new source/bin/log receipts before independent review. See [workflow](../FORK-WORKFLOW.md).
