# Opt-in C4 historical confidence governor

Base: `806939db941f4ddb9bff4961c9dc5cd6e0df7798`. This is a DSpark-inspired
**historical acceptance proxy**, not a port of DeepSeek's trained Confidence
Head, model weights, or throughput estimator.

## Enable and rollback

Keep the existing C4/Q4 streaming configuration, and opt in with:

```sh
export STRATA_BATCH_GENERALIZED=1
export STRATA_BATCH_SLOT_SOLO=1
export STRATA_BATCH_ROW_TABLE=1
export STRATA_BATCH_GRAPH_LIMIT=8
export STRATA_BATCH_DSPARK=1
export STRATA_BATCH_AUDIT=/absolute/new-run/window.audit.jsonl
# Existing model/serve/MTP arguments remain required:
# --batch 4 --batch-rows 12 --spec 8 --batch-spec 8
# --max-context 262144 --kv q4_0 --kv-resident 20480 --batch-expert-merge
```

`STRATA_BATCH_DSPARK` accepts only `0` or `1`; absent/`0` keeps the old path.
The old maximum `batch-spec=6` remains unchanged when disabled. With the feature
on, the maximum is explicitly configured by `min(spec, batch-spec)`, at most
**8 verification rows = 1 pending token + 7 proposals**. Keeping `spec=6` and
`batch-spec=6` does not silently grow to eight. Suffix drafting is rejected in
this initial feature. Generalized batching's existing no-padding, single-stage,
text/NVIDIA/Q4/audit restrictions still apply.

Rollback: unset/disable the feature and restore `--spec 6 --batch-spec 6` if S8
was configured. No state/quantization format or sampler migration is needed.

## Causal contract

Each canonical owner has one fixed-size `DsparkConfidence` instance. A BGEN
admission resets it to the new native audit request generation, including hot
slot reuse and checkpoint rewind. Nothing is learned across requests.

For owner verification step **N**, the planner sees only completed outcomes
through **N-2**. The latest completed step N-1 is quarantined. The ring holds
16 eligible windows plus one quarantined window. It is queried after step N-1
has committed to prepare the next private draft count, and the same decision
is used when step N starts. Current draft probabilities/tokens and current
logits are not confidence inputs.

Only finite, committed, continuing owner windows are observed, after the
existing audit accounting has succeeded. EOS/cancel/length terminal windows
and nonfinite owners cannot seed a later request. Duplicate/out-of-order
window IDs, wrong generations and invalid prefix sizes fail closed.

For proposal position d, a historical conditional trial exists only when all
earlier proposals matched. Rejected suffixes and unverified rows are censored.
The planner applies Beta(1,1) smoothing to these conditional rates; unseen
positions use the pooled historical estimate. It chooses the deepest prefix
whose cumulative survival is at least 0.5, with a two-row lower bound (unless
only a pending row fits), and a cold depth of three until four eligible
windows exist. The 16-window horizon permits recovery after a low-acceptance
semantic region; a fully accepted chain can reach the configured S8 limit.

This is an experimental policy, **not a promised speedup**. A future hardware
cost model or trained confidence head is outside this PR.

## Controller and memory contract

- The old per-slot `row_budget / active` ceiling is bypassed only for DSPark's
  demand calculation. Existing fair allocation and segment packing still give
  every active owner its pending row and enforce **12 total rows**, individual
  segments <=8 and two complete causal dense groups <=8 each.
- At high confidence, one active owner may get 8, two get 6+6, three 4+4+4,
  and four 3+3+3+3. Mixed confidence can redistribute spare rows; low demand
  need not fill all twelve.
- Next private drafts are limited by the already planned next allocation.
  When an admission changes the active set, unused private proposals are
  simply not verified/committed. The controller tracks generated capacity,
  never reads the zero-filled unused suffix as a real proposal.
- In this mode alone, draft generation runs to that planned count with its
  current-probability early-stop threshold disabled (`min_p=0`). The current
  proposal-probability width gate and the old immediate `row_limit` updater
  are bypassed. Leaving them active would let newer/current information
  override the N-2 policy. The request's actual target sampling parameters
  and random counters are unchanged.
- Exactly one active decode owner still uses the existing
  `ScopedSoloBinding`; multi-owner windows still use the device row table.
  Sampling and accepted-prefix commit call sites are unchanged.
- Physical verifier/layer strides stay **12**, Q4 Host/map/resident storage
  stays unchanged, and no second verifier/GPU scratch arena is allocated.
  The host history is under 512 bytes per owner. Existing Solo/batch LRUs
  remain independent; S8 graph/memory pressure requires real GPU acceptance.

## Independent diagnostics; V1 unchanged

No common audit structure, conservation enum, device table ABI or V1 wire
field changes. READY advertises `batch_dspark=1 batch_dspark_lag=2
batch_dspark_history=16` only when enabled.

With `--batch-trace`, stderr additionally emits one independently parseable
`batch-dspark-dispatch {JSON}` line per successful window. It contains only
window/slot/generation/step identifiers, the eligible history watermark,
observation count, historical observed rate and requested/allocated rows;
no token IDs, prompt text or draft probabilities. The controller clock is
monotonic across idle periods; correlate using request generation and the
window trace, not an assumption that unrelated admission records have the
same sequence number.

## Local checks and serial acceptance

```sh
c++ -std=c++20 -O2 -DNDEBUG -Iinclude tests/core/dspark_confidence_cpu.cpp -o /tmp/dspark
/tmp/dspark
c++ -std=c++20 -O1 -g -DNDEBUG -fsanitize=address,undefined \
    -fno-omit-frame-pointer -Iinclude tests/core/dspark_confidence_cpu.cpp -o /tmp/dspark-asan
/tmp/dspark-asan
```

The public test checks N-2 noninterference, exact 16+1-ring/reference agreement,
censored trials, owner/generation reset, invalid inputs, online S8 growth,
retreat/recovery and production fair/pack geometry. It uses runtime checks,
not disabled Release assertions. CMake target/CTest: `dspark_confidence_cpu`.
The existing CPU-contract CI also runs it.

Local results (Apple Silicon host; no native CUDA execution):

| Check | Result |
| --- | --- |
| New helper, Release `-DNDEBUG` and ASan/UBSan | 2,137,922 runtime checks each, pass |
| Existing cache/layout/Solo/MTP/packing/scratch/audit/planning CPU regressions | 11 sources, Release and ASan/UBSan pass |
| Position helper, Release and ASan/UBSan; conversation-file format | Pass |
| No-CUDA/no-HIP CMake Release target and CTest | 1/1 pass |
| Deliberate N-1 mutation outside the source tree | Rejected, exit 1 as required |
| Server mock regressions | 966 tests, 951 pass, 15 skipped |
| Analyzer/tokenizer mocks | 49 tests, 46 pass, 3 skipped |
| NVFP4 ready-pack mocks | 4/4 pass |
| Controller host syntax; unchanged target/commit/audit source comparison | Pass, not runtime parity |

The first broad mock run exceeded a 240-second harness budget; a declared
600-second run completed in 378 seconds. Analyzer imports required the CI's
`PYTHONPATH=tools:.`. The full NVFP4 setup suite is not qualified locally:
seven unchanged installer mocks follow the native macOS setup redirect and
fail looking for a temporary `metal/setup_mac.py`. Their source and setup
helpers match the base exactly; the existing Ubuntu CI remains the authority
for that gate. No baseline tests were weakened to accommodate the host.

Controller host syntax with Metal compatibility headers is not CUDA
compilation or runtime qualification. Before merging into a production
configuration, the authorized serial bench must:

1. Build the frozen source on sm120; validate flag refusal and explicit S8.
2. Compare DSPark off/on at the same expert mapping and sampling, with GPU
   GDN/conv/PLE/QSA and target sampling parity, including T1..8 and prefixes.
3. Run same-PID `1->4->1->4->1->4`, all owners, reuse/rewind/EOS/cancel and
   non-greedy/penalty histories; prove no duplicate commit or stale generation.
4. Verify all history watermarks are causally old, V1 conservation is intact,
   and minimum GPU free remains >=300 MiB. Separate diagnostic trace runs
   from throughput runs if trace overhead matters.
5. Report acceptance length, verified rows, graph misses, cold/steady timings
   and actual routed occurrences. No tok/s or math-parity success is implied
   by this PR's CPU tests.
