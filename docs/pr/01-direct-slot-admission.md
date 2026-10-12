# PR-1 · Direct Slot Admission and lazy RAM spill

- Branch: `pr/01-direct-slot-admission`.
- Execution base: accepted PR-2a SHA; currently unfrozen/pending.
- Goal: use two canonical target/draft slots directly; eliminate the third solo allocation and hot GEN1→slot full-KV copy while retaining conversation isolation.
- Scope: admission/state ownership in `src/program/generate.cpp`, Prefill slot binding in `include/strata/prefill/prefill.hpp` / `src/prefill/prefill.cpp`, owner/private MTP state, service admission/completion in `serve/server.py` and slot-only tests; required routed/raw-gap accounting travels with the behavior.
- Required behavior: direct owner-bound prefill/resume, spill to RAM only when displaced, verified park/restore provenance, no cross-slot residual reuse, queue/cancel/nonfinite/BDONE lock cleanup.
- Non-goals: changing GGUF/model profile, unsafe global KV sharing, adaptive12/Narrow/head/QSA, optional host timing diagnostics or C2 streaming.
- Contracts: allocation ledger proves only two canonical targets; trace proves absence of full admission copy; single/dual/queued4, park/restore/reject/cancel/healthy-after-poison tests. Require exact routed hit accounting and raw gaps rather than legacy proxies.
- Attribution: historical one-run 168.96 tok/s / legacy hit95.731% / free293 MiB is not current performance qualification. Removing one third state is not equivalent to recovering the entire official C1→C2 5,010 MiB cache delta. No isolated speed percentage is established for this PR.
- Acceptance: new true-C1 control and fresh C2 quality/cache/headroom gates; do not assume an allocation saving automatically raises end-to-end speed.
- Rollback: explicit allocation/admission feature unit; dependent adaptive features must remain compatible or be reverted first.

Freeze exact base and writer; deliver new build/contract/benchmark hashes and a real PR URL without self-merging. See [workflow](../FORK-WORKFLOW.md).
