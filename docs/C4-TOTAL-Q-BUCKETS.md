# C4 bounded Total-Q template buckets (PR-3)

Base: `806939db941f4ddb9bff4961c9dc5cd6e0df7798`.
Branch: `feat/deepseek-pr3-total-q-buckets`.

## Status and explicit trade-off

Experimental, **OFF by default**. This is four bucket families with **eight exact
G2 target/commit pairs**, not four padded graphs and not a proof of GPU speedup.
The existing 12-row arena, dense two-stream controller, Q4 streaming, canonical
state owners, target sampler and accepted-prefix commit are unchanged.

To bound the graph family without padding or splitting a causal segment, this
policy may **shorten optional draft prefixes**. Every active owner retains its
pending input. It chooses the largest feasible declared template, with a rotating
deterministic preference between equal-row allocations. A trained confidence
head or a new sampler is not introduced. This is a post-filter for an upstream
row/cost/confidence planner, not a replacement for it.

Some asymmetric demands lose substantial offered work: e.g. `[1,1,6,1]` may
fall from nine offered rows to the four pending rows. Therefore a reduction in
capture misses is **not** sufficient for promotion: throughput and accepted
length must be measured. Leave this OFF if the extra decode steps cost more than
the avoided captures. No production default or acceptance claim changes here.

## Why rows and cut remain real key data

R12 `6+6` requires a legal cut at six, while `4+4+4` requires a cut at eight.
A family-only key would reuse the wrong captured dense topology. Negative owner
IDs are also invalid descriptors, not a padding API. This implementation never
adds dummy rows, grows a proposal, changes token order, or uses a 16-row arena.

| Family | Exact rows/cut | Capture-only owner-width witness |
|---|---|---|
| Q3 | 2 / 1 | 1+1 |
| Q3 | 3 / 1 | 1+2 |
| Q3 | 3 / 2 | 2+1 |
| Q6 | 4 / 2 | 1+1+1+1 |
| Q6 | 6 / 3 | 3+3 |
| Q9 | 9 / 6 | 3+3+3 |
| Q12 | 12 / 6 | 6+6 |
| Q12 | 12 / 8 | 4+4+4 |

The fully loaded golden layouts `3+3+3+3`, `6+6`, and `4+4+4` are preserved
exactly. For any two, three or four active owners, pending-only widths reach a
legal R2/c1, R3/c1 or R4/c2 fallback. All selected cuts are computed with the
production `segment_group_boundary`; no segment crosses the dense cut.

The key prefix is `[-3, family, actual_rows, cut]`, followed by the existing
execution-profile fields. Actual rows cannot be omitted: R2/c1 and R3/c1 are in
the same Q3 family. Owner IDs, segment count and request generation remain
runtime descriptor contents, as in the baseline device-table path.

## Integration and G2 invariant

`plan_total_q_buckets` is applied after the existing allocation/pack stage and
before token rows are constructed. It only shrinks those allocated prefixes.
Single-owner Solo and single-owner/prefill admission paths bypass this policy.

The verifier requires dense/narrow parallelism, the existing shared-after-ring
path, merged experts and one frozen device execution profile. Missing or changed
profiles fail closed; there is no serial fallback. The entire
`Verifier::record_window` body, including the merged G2 fork/join controller and
all arithmetic kernels, is byte-for-byte unchanged from the base.

No speculative target or fake checkpoint is executed at startup. READY prewarm
only captures and uploads all eight target/commit pairs using synthetic metadata.
It restores host staging and generation/sequence counters before admitting real
requests; audit request generations are never advanced by prewarm.

## Graph retention and memory accounting

The eight exact template pairs are pinned. Legacy admission and single-owner
row-table graphs retain their own bounded LRU using `STRATA_BATCH_GRAPH_LIMIT`.
Both ordinary LRU eviction and OOM recovery skip pinned keys. A missing sealed
template is an error, **not** a silent recapture disguised as a cache hit.

With `GRAPH_LIMIT=8`, the maximum is eight pinned pairs plus eight ordinary
pairs. This can retain more memory than the baseline's single eight-pair LRU;
it bounds and separates families rather than promising fewer resident bytes.
If Solo is also enabled, its independent eight-target/four-commit bank remains
additional. The combined graph-exec count can reach 44; these are not duplicated
verifier arenas. Driver allocations need real GPU measurement.

New-mode graph instantiation checks resident free memory is at least 300 MiB;
READY requires at least 512 MiB after the complete bank is captured. No memory
query is added to hot replay. These queries do not prove a transient peak bound:
continuous NVML/min-free observation is still required in GPU acceptance.
No expert capacity or reserve is silently reduced to accommodate the bank.

## Enable in an isolated experiment

Keep the existing model/assets/CLI and all accepted C4 settings. Add:

```sh
export STRATA_BATCH_GENERALIZED=1
export STRATA_BATCH_ROW_TABLE=1
export STRATA_BATCH_TOTAL_Q_BUCKETS=1
export STRATA_BATCH_GRAPH_LIMIT=8
export STRATA_BATCH_DENSE_PARALLEL=1
export STRATA_BATCH_NARROW_PARALLEL=1
export STRATA_BATCH_SHARED_AFTER_RING=1
# Retain Solo in the unified configuration, if enabled in the accepted baseline:
export STRATA_BATCH_SLOT_SOLO=1
# Use a fresh exclusive path on EVERY new engine run:
export STRATA_BATCH_AUDIT=/owned/new-run/window.audit.jsonl
```

CLI still includes `--batch 4 --spec 6 --batch-spec 6 --batch-rows 12
--max-context 262144 --kv q4_0 --kv-resident 20480 --batch-mtp
--batch-expert-merge`, plus the existing model/runtime options.
Malformed flags, ROW_TABLE=0, missing batch MTP, insufficient graph limit,
unsupported backend/profile or insufficient free memory are errors.

## Audit evidence

Header capabilities expose `total_q_buckets` and `total_q_template_pairs=8`.
Multi-owner windows retain `execution_path=batch_rows` and
`graph_key_kind=rows_cut`; `row_table_bucket` additionally records family,
exact template ID, offered rows and retained rows. This exposes the cost of
prefix shortening rather than hiding it in a cache statistic.

The native writer rejects a multi-owner new-mode window without a declared
exact template, cache hit, G2 parallelism or shared-after-ring. OFF and Solo
records retain the existing behavior and serialize `row_table_bucket=null`.

## CPU validation and mutation evidence

- `total_q_buckets_cpu`: 737,895 checks; 6,588 valid dynamic plans including
  sparse owner sets, widths through eight, all rotations and all owner
  permutations, using the production Window/Commit ABI. 5,556 of these uniform
  synthetic cases shorten prefixes; this is not a production frequency claim.
- A key mutation omitting actual rows fails with
  `actual rows must not collide within one bucket/cut` (exit 1).
- Tests exercise the production pin classifier/count helper across 1,000
  admission-cache insertions, a bounded ordinary LRU and protected-bank OOM
  behavior. This is a CPU cache model, not actual CUDA execution.
- `total_q_buckets_audit_cpu`: 163 checks; all eight serialized templates,
  ten fail-closed dispatch mutations and OFF compatibility.
- Both tests use active checks under `-DNDEBUG`; Release and ASan/UBSan pass.
- Existing descriptor, Solo, row/scratch, MTP, audit, cache, telemetry and QSA
  guard CPU regressions are also run. CMake/CTest targets and GitHub CPU CI are
  wired. Host Metal-compatibility syntax checks are not a CUDA build.

## Serial acceptance before promotion

1. Compile the exact PR head on CUDA/sm120. Run existing public GDN/PLE/QSA
   table GPU parity fixtures and complete model startup with all eight captures.
2. Verify real cache hits after READY across owner/generation changes and
   admission churn, and assert G2/shared-after-ring on every multi-owner window.
3. Compare leaf/model math at identical selected rows/cut. The allocator changes
   offered work, not the target sampler; do not relax exact sampling or state
   parity to make an experiment pass.
4. Run same-PID `1 -> 4 -> 1 -> 4 -> 1 -> 4`, three rounds per load, including
   greedy/non-greedy, reuse/rewind/cancel and sparse owners. Preserve Q4 Host/map
   streaming and all four 256K logical capacities.
5. Measure offered/retained rows, accepted output length, capture/steady CPU/GPU
   time, expert occurrence hit rate and continuous minimum free memory >=300 MiB.
   Compare identical expert capacity/reserve/fixtures against OFF. Reject a speed
   regression even when the warmed graph-hit percentage is 100%.

No CUDA compilation, GPU parity, real graph-memory bound, model quality result
or throughput gain is claimed by local CPU validation.
