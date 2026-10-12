# PR-7 · Optional host/ring accounting

- Branch: `pr/07-host-ring-diagnostics`.
- Execution base: exact accepted runtime-interface integration SHA; currently pending.
- Goal: close host phases in integer nanoseconds and separately attribute ring sequence wait, callbacks, activation quantization, plan/jobs and native pool work.
- Scope: optional host-profile types/TLS, ring/verifier/CPU-pool instrumentation, analyzer and analyzer tests. No performance policy or arithmetic change.
- Required behavior: monotonic clock, integer closure, profile defaults OFF; thread-local/reentrant ownership and nested phase accounting. Do not compare diagnostic runs as official throughput scores.
- Non-goals: enabling device plans that break Dense parallelism, speculative prefetch changes, optimizing a phase merely because it has a large label, or treating wait/work intervals as disjoint hardware cycles.
- Contracts: exact closure for every window, phase sums/counters, profile ON/OFF numerical behavior, zero/stale/missing trace rejection, TLS isolation and real executed analyzer tests.
- Interpretation: sequence host-wait is not pure GPU execution; host_work can overlap GPU/worker activity. Unknown bandwidth/occupancy/clock states remain unknown. Diagnostic historical decomposition is not a speed claim.
- Acceptance: approved short diagnostic run at exact SHA plus unchanged C1/C2 behavior; scoring runs leave profiling disabled.
- Rollback: optional observability unit only, without changing runtime feature selection.

Freeze interface/base/head SHA, writer and tests before implementation; attach integer closure and logs. See [workflow](../FORK-WORKFLOW.md).
