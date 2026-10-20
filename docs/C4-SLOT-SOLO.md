# Experimental C4 Solo-in-slot (Commit A)

Base: `8323d0f8c82908ec5077287a7b6241780ca3362c`. This is an opt-in CUDA experiment, **not** a qualified speedup. Device tables / rows-only graph keys (Commits B/C) are not implemented.

## Scope and switch

`STRATA_BATCH_SLOT_SOLO` accepts exactly `0` or `1`; unset means `0`. ON requires the existing generalized NVIDIA Flash-Next path, four canonical target owners, 12 physical verifier rows, Q4 KV, logical context 262144, resident 20480, MTP and spec <=8. The existing batch-spec validation still applies (depth6 requires rows12). Images, fixed padded layouts, multiple stages, asynchronous adaptation and grow remain outside the generalized contract. No expert, PCIe, drafter, suffix, precision or sampling policy is changed by this flag.

Use the **same accepted C4 configuration and same binary** for OFF/ON; change only this flag and unique log paths. In particular retain native arguments `--max-context 262144 --kv-resident 20480 --kv q4_0 --spec 6 --batch-spec 6 --batch-rows 12 --pipeline-windows 0`, `STRATA_BATCH_GENERALIZED=1` and the qualified reserve/graph limit. The server/parallel launcher supplies `--batch 4` once; do not duplicate it in cfg. Do not add the legacy raw `--batch-slot-only` flag to C4: generalized ownership is already canonical internally. `--vram-reserve-mib` is the real reserve option, not `--reserve`.

## Execution and ownership

- `fast_batch_step` uses actual active owners and allocated rows. Exactly one active text owner, 1..8 rows and **no prefill/admission chunk work** selects Solo for one target window. All other windows retain the existing segmented batch path. No request is moved to GEN.
- Dispatcher-local `ScopedSoloBinding` spans target run, common per-row sampling, accepted-prefix judgment and commit. It swaps graph handles and the `ss_` pointer, **not SessionState contents**. Target state and private MTP stay at their canonical owner; one existing 12-row verifier/arena is reused.
- Each of four owners has a noncopyable parked graph bank. Binding/unbinding fences outstanding target/commit/copy/side-stream work. A failed consumed window is fatal, never retried through the legacy path. If CUDA cannot drain, the terminal binding is deliberately not restored; verifier teardown owns both banks.
- Target execution suppresses `run()`'s global re-sampler, then uses the existing batch row sampler exactly once with private slot params/history and absolute-position Philox counter. The shared skip predicate no longer drops greedy logit bias. The baseline **still rejects logit_bias on batching requests**; this patch does not add per-slot bias support.
- GDN and QSA commit use logical `min(physical_rows, 8)` but retain **12-row per-layer strides**. Single-row self-commit is disabled while bound, so accepted-prefix state and PLE predecessors advance exactly once through the controller's commit.
- Owner/address/static profile changes invalidate that owner's graphs after draining and increment its address epoch. Normal generation reuse/park/restore at unchanged addresses does not increment the epoch. Epoch is not request_generation.

## Graph memory

After all static setters and before INFO/READY, ON captures one maximum-width target and one commit per owner; it does not execute target graphs or warm every shape/variant. Subsequent on-demand target shapes and both residency variants share a global Solo LRU of `min(max(STRATA_BATCH_GRAPH_LIMIT,1),8)` target executables. The existing configured default remains four; an explicit limit of eight now retains eight rather than being silently truncated to four. Up to four owner commit graphs are additional. The existing legacy layout LRU is separate. These are **graph counts, not a guaranteed MiB budget**; there are no extra target sessions/KV/weights/scratch arenas.

Startup logs actual prewarm free-before/free-after/retained-byte delta and refuses READY below 512 MiB. It does not silently lower expert capacity. Bench must qualify the entire dynamic-layout campaign with its >=300 MiB watchdog; a successful prewarm does not prove future driver allocation peaks are safe. Both arms retain identical reserve/cache capacity for attribution. Trace diagnosis and trace-free performance remain separate.

## Native evidence

`INFO batch_slot_solo=0|1` states the configured capability. WindowAuditV1 retains all scopes, generations, row owners, routed/hits/cpu/offload/skipped integers and actual BT emission counts. Added `dispatch` fields:

|Field|Type / value|
|---|---|
|execution_path|string: `legacy` or `slot_solo`|
|graph_key_kind|string: `legacy_layout` or `slot_solo` (NOT rows-only)|
|graph_key_rows|integer: actual target rows|
|graph_owner_slot|integer: native owner on Solo, -1 on legacy|
|graph_address_epoch|unsigned integer: owner static-address/profile epoch on Solo, 0 on legacy|

Header capabilities add `slot_solo` (bool), `row_table=false`, `slot_solo_target_graph_limit` (integer; 0 when OFF), `slot_solo_commit_graph_limit` (4 when ON, otherwise 0). `dense_parallel` and merged remain false on Solo. Head/resolve metadata comes from the actual owner/shape/residency graph; no batched-head flag is inferred from env eligibility. The ordinary Solo attention entry uses scalar attention rows plus union/scalar resolve counts, not a falsely attributed QSA tile count.

Legacy detailed `batch-host` / ring phase traces are instrumented in `run_slot_rows()`, not Solo `run()`. Their bins are cleared and those two detailed trace records are **suppressed for Solo** instead of reusing a prior legacy window or inventing phase timings. WindowAuditV1, batch trace summary, native route/emission counts and external request timing remain available. Do not interpret absent detailed Solo phase bins as zero host work.

## Qualification gates

CPU: strict flags/dispatch geometry including prefill exclusion, noncopyable graph-bank ownership and arbitrary-owner exchange/restore, cap/stride boundary, sampler predicate, audit serialization, existing planner/scheduler/server/tokenizer regressions. Host syntax or ASan success is **not CUDA or GPU qualification**.

Before speed claims, bench must build the delivered SHA on CUDA13.3/sm120, then compare OFF/ON with identical expert/reserve/PCIe/spec/suffix/model/IDs and audit boundaries. Required GPU checks: all four owners, T=1..6 and each accepted prefix; exact target logits (where exposed), emitted IDs, GDN/conv, pooled index/tail, encoded Q4 KV and PLE state; private MTP continuation; greedy and non-greedy/penalty cases; unchanged batch-bias rejection; cold/warm/EOS/cancel/reuse/park/restore; 1->4->1 and arbitrary-owner Solo; no NaN/stale callbacks/state bleed. Verify native scopes and HTTP generations, graph ownership/limits/memory throughout, and C4 multi-active regression before >=6 paired trace-free throughput trials. No 200+ tok/s claim is made here.
