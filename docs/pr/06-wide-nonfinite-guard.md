# PR-6 · Capacity-safe NaN/Inf guard

- Branch: `pr/06-wide-nonfinite-guard`.
- Prepared base: `7695859aa13afb248bdb4d36f9b2e95707db079d` (official v0.1.43).
- Status: task prepared; no runtime patch or new qualification yet.
- Goal: make allocation, row-mask consumption and nonfinite lifecycle safe for verifier capacity 9–12 (and tested 16), without disabling the guard.
- Read/write scope: `include/strata/core/verify.hpp`, `src/core/verify.cpp`, relevant nonfinite consumers in `src/program/generate.cpp`, `tests/core/nonfinite_wide_*`, scoped CMake registration. If the C2 consumer arrives later, add its caller test atomically in that PR.
- Required behavior: allocate mapped host/device flags from `max_t * kNonfiniteBlocks`, initialize all allocated rows, mask only actual rows within capacity, invalidate stale flags, preserve poisoned-slot termination and healthy-slot progress. Retain upstream controls and NaN/Inf defaults.
- Non-goals: globally raising legacy `kVerifyMaxT=8`, enabling adaptive12, modifying expert/head arithmetic, disabling diagnostics to hide failure.
- Contracts: CPU masks/capacity/invalid boundaries; GPU direct and captured graph tests for NaN/+Inf/−Inf, finite/stale rows, capacities 1/8/9/10/11/12/16 and vocab 17/248320/248321; exact mapped canary bounds. The historical fused suite executed 3,618 cases, not a receipt for this new branch.
- Integration gates: valid C1 behavior unchanged; nonfinite/cancel/BDONE cleanup releases locks and suppresses poisoned emissions. New wide consumers must use the verified per-row API (`nonfinite_in`) rather than inventing another accessor.
- Rollback: revert this isolated safety change only if no dependent wide feature remains enabled. All feature PRs must declare this dependency.

Execution must fill the head SHA, commands/counts/exits and fresh receipts before review; see [workflow](../FORK-WORKFLOW.md).
