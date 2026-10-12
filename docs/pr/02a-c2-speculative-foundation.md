# PR-2a · Minimal speculative C2 foundation

- Branch: `pr/02a-c2-speculative-foundation`.
- Execution base: accepted PR-6 SHA; currently unfrozen/pending.
- Goal: port the minimal PR #1524-era speculative C2/private-slot foundation onto v43 so later optimization PRs are independently buildable.
- Scope: batch segment helper/types, `mtp.hpp/.cpp`, `verify.hpp/.cpp`, batch-MTP callers in `generate.cpp`, segmented/GDN tests and scoped CMake entries; service changes only where a concrete lifecycle interface requires them.
- Required behavior: slot-private drafts and residuals, ordered append and accepted-prefix state, ≤8-row verification, upstream v43 nonfinite/cancel/BDONE cleanup and ordinary C1 MTP behavior.
- Non-goals: 12-row capacity enablement, Direct Slot allocation change, lazy RAM spill, Narrow Dense, wide head, QSA Tile or optional profile ABI. This is compatibility infrastructure, not a standalone speed gain.
- Contracts: CPU layout/token-slot mapping, partial-prefix commit and private-state poison tests; GPU segmented GDN parity (historical suite: 72 layouts / 882 commits); server queue/cancel/resume/slot isolation and true C1 regression. New source requires new counts/receipts.
- Risks: donor code predates v43 lifecycle and nonfinite fixes; do not overwrite those with old whole-file copies. Caller and state interface changes stay atomic.
- Rollback: revert foundation only after dependent runtime features are disabled/reverted.

Freeze exact predecessor SHA and writer before execution. Record source/test/bin/log hashes; no empty PR or skipped test counts as implementation. See [workflow](../FORK-WORKFLOW.md).
