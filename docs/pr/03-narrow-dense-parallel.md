# PR-3 · Narrow Dense parallelism

- Branch: `pr/03-narrow-dense-parallel`.
- Execution base: accepted PR-2b SHA; currently pending.
- Goal: make dual-slot windows ≤8 rows use the existing two-stream dense topology with private scratch and correct joins, rather than reverting to a serial narrow path.
- Scope: verifier graph/layout/stream dispatch in `verify.hpp/.cpp`, minimal generate metadata consumers, dense/GDN/private-state contracts. Keep wide behavior and head/QSA primitives unchanged.
- Required behavior: two actual dense groups, private state/scratch by slot, `shared_after_ring` dependency, no premature reuse or cross-stream poisoning. Single-slot/boundary windows safely fall back.
- Non-goals: STRATA_VERIFY_DEVICE_PLAN changes that break merged dense parallelism, expert arithmetic, wider head, QSA kernel changes or streaming.
- Contracts: segmented GDN/private-state parity, dual-stream poison/replay, narrow/merged join ordering and fallback; native ON `dense_groups/dense_parallel/shared_after_ring=2/1/1` versus OFF `1/0/0` coverage, actual row distribution and gaps.
- Historical evidence:189.77 versus183.84 pooled (+3.23%); pairs +2.834/+1.050/+5.794%. Host sequence-wait reduction8.44–11.26% is not pure GPU kernel timing or proof of three significant wins.
- New acceptance: exact new branch CPU/GPU receipts, true C1 control and three same-binary ON/OFF queued4 pairs without changing frozen assets/spec/reserve.
- Rollback: narrow-layout selector/stream unit; wide merged topology remains unchanged.

Freeze base/head SHA and GPU lease; actual executed counts and exits are mandatory. See [workflow](../FORK-WORKFLOW.md).
