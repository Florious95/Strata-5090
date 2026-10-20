# Experimental C4 device-row table

This is an opt-in CUDA experiment for Qwen3.8-Flash-Next, not a default change or a measured speedup. It builds on Solo-in-slot (PR #12). It does not implement a row-only graph or single-slot depth eight.

## Run boundary

Set `STRATA_BATCH_ROW_TABLE=1` only with audited generalized C4, physical `--batch-rows 12`, Q4 KV, `--max-context 262144`, `--kv-resident 20480`, merged experts and the existing single-stage variable-segment scheduler. Keep the existing spec/adaptive policy. `STRATA_BATCH_AUDIT` must name a fresh private file. The server/parallel launcher supplies `--batch 4`; do not duplicate it in configuration args.

The switch accepts only `0` or `1`, defaults to `0`, and is independent of `STRATA_BATCH_SLOT_SOLO`. Admission uses its existing canonical-owner path; the table is used only for segmented target windows. CUDA is required. HIP, Metal, SYCL, image histories, KV grow, different resident sizes and replaced state addresses are not qualified. The initial kernel profile requires the normal GDN recurrence/commit and multi-score kernels: do not enable `STRATA_GDN_SPLIT=1`, `STRATA_GDN_COMMIT_SPLIT=0`, `STRATA_DBG_GDN=1`, `STRATA_QSA_ROWS=1` or score-halving, or disable multi-scores. Unsupported profiles fail startup instead of silently substituting a different policy.

Use the same IQ3_XXS pack, MTP/PLE assets, affinity/clocks, tokens, sampling, reserve, expert-cache byte budget and warmup on both arms. Trace diagnostics and clean performance runs remain separate. The real reserve option is `--vram-reserve-mib`. A new port alone does not isolate GPU memory.

## What the key means

A segmented graph is keyed by `(R, cut, frozen execution profile)`, not owner IDs, owner order, segment count, request generation or accepted-prefix lengths. The key is reported as `rows_cut`, never `rows_only`.

* R12 / cut6 can replay both owners 0,3 with widths 6,6 and owners 3,1,0,2 with widths 3,3,3,3.
* Widths 4,4,4 require cut8. Cut6 bisects a causal segment and is rejected.
* Both dense groups remain at most eight rows. Physical verifier scratch remains twelve rows.
* The CUDA capture uses fixed row/group bounds and fixed four-segment capacity. Device kernels select each current owner; inactive segment CTAs are no-ops.

`STRATA_BATCH_GRAPH_LIMIT` remains an LRU count, not a MiB budget. Admission graphs and target families can still cause eviction. This change removes owner-layout permutations from the new target key; it does not guarantee zero recaptures or a fixed graph-memory footprint.

## State and address lifetime

`DeviceSlotTable` is non-owning. GDN, PLE, QSA pools, complete Q4 Host K/V, CLOCK-map arrays, indexer state, RoPE pointers and indexer-tail snapshots still belong to their canonical slots. No new SessionState, KV pool, weights or verifier arena is created. `DeviceQsaPools` borrows writable Q4 pointers directly from `QsaState`; append and Host-to-VRAM resolve require that writer capability. Attention receives only its explicit `read_only()` view. A const descriptor freezes pointer identities, not the owned cache contents. No const reader is cast back to a writer.

The ABI payload is 14,400 bytes: table 14,016 + window 352 + commit 32. The requested device allocation is 14,416 bytes including epoch/control alignment; separate mapped status and staging request 388 pinned bytes (not a claim about OS page-rounded physical allocation). Target and commit graph storage is additional and must be measured.

The table is sealed after final head/cache/split setters, before READY. Host staging and device descriptor addresses stay fixed. The target graph uploads the current window; commit uploads the accepted prefixes. Both carry sequence and table epoch; segments carry independent request generation and owner-address epoch. Request generations come from native admission, not HTTP guesses. Static owner/profile changes fail closed and require restart; live table replacement is not implemented.

The old arithmetic bodies serve the legacy and table entries. GDN commit accepts one prefix per whole segment. PLE reuses its existing per-row history snapshot as working history and later selects the accepted snapshot. This introduces history copies but no extra history arena. QSA appends write authoritative Host K/V and resident pages, scores use each owner's keys, each owner's complete selected-page union resolves before attention, and attention loads that row's pools. Rejected drafts retain the existing overwrite/repair contract. No CPU ownership copying or asynchronous commit finisher is introduced.

Device descriptor failures make a consumed window fatal before output collection/commit publication. There is no legacy retry of partially executed work.

## Merged shared-stream capture boundary

The merged controller publishes its complete expert window outside the per-group routing branch. Therefore it must not select the legacy late shared-expert fork: that fork is emitted only by the non-merged publication path. In a one-row table window with shared streaming enabled, skipping the early fork would leave the shared stream outside capture; joining its event can invalidate capture, with the later MoE combine reporting only the previous error. Merged windows now take the existing early-fork path. Non-merged late/early policy and all arithmetic bodies are unchanged. Shared join record/wait failures are checked at their source instead of surfacing at combine. A CPU policy regression covers all eight requested-late / effective-all-resident / merged combinations. The asset-free `batch_shared_stream_gpu` target first requires the old missing-fork topology to reject capture, ends that invalid capture safely, then plans eight production-helper captures and 40 dynamic-value replays across real main/shared streams. It is a dedicated topology gate, not full-model arithmetic or performance qualification.

## WindowAuditV1 additions

Existing route integers, scopes, lifecycle, HTTP slot/generation identity and resolver categories are unchanged. Additions:

* Header `capabilities.row_table`: boolean, true only when configured.
* New table window `execution_path="batch_rows"`, `graph_key_kind="rows_cut"`, `graph_owner_slot=-1`, positive `graph_address_epoch` (table epoch).
* `graph_key_cut`: integer, 0 for one group or the complete-segment cut.
* `graph_key_profile`: integer, `0x100 | ar_off | (merged<<1) | (dense_parallel<<2) | (shared_after_ring<<3) | (generalized<<4)`.
* `row_table_qsa_calls`: integer map with exactly `score_launches`, `score_segments`, `attention_launches`, `attention_rows`.

For a complete 48-layer window, with 12 QSA layers, G dense groups, N complete segments and R rows:

```json
{"execution_path":"batch_rows","graph_key_kind":"rows_cut","graph_key_rows":12,"graph_key_cut":6,"graph_key_profile":278,"graph_owner_slot":-1,"graph_address_epoch":1,"qsa_tile_calls":0,"qsa_tile_rows":0,"qsa_scalar_rows":0,"row_table_qsa_calls":{"score_launches":24,"score_segments":48,"attention_launches":48,"attention_rows":144}}
```

Here score launches = 12G actual CUDA launches; score segments = 12N logical owner-segment score bodies (not CTA count); attention launches = 24G actual chunk + merge CUDA launches; attention rows = 12R query rows, not doubled for the merge. `kv_resolve_calls` still counts per-owner resolver-body invocations: width one is `scalar_per_row`, width >1 is `union_per_slot`, not the number of CUDA grid launches. Inactive capacity contributes zero.

Legacy same-slot tile counters are zero on the new table path. They must not be used to pretend the old API was called, nor to reject a table run solely because they are zero. The new counters are zero for legacy/Solo windows. `graph_key_cut/profile` are zero there. Solo owner/epoch semantics do not change.

## Required evidence before qualification

Passed CPU ABI/layout/accounting checks are not GPU parity. Compile CUDA 13.3 / sm120; run dedicated synthetic table tests and existing GPU contracts, including full Q4 Host/map/union attention, all owner permutations, changed segment counts at the same key, accepted prefixes, inactive-owner poison and graph replays. Compare OFF/ON output token bits and post-commit canonical state, not only tokens. Also compare the default OFF build against the parent where possible: sharing a factored arithmetic body is not by itself historical-binary parity.

Dedicated test targets (written, not a claim of GPU execution):

```sh
cmake --build build --target batch_device_table_gpu batch_device_qsa_gpu batch_shared_stream_gpu
ctest --test-dir build --output-on-failure -R '^batch_(device_(table|qsa)|shared_stream)_gpu$'
```

All three must really execute and return zero; a return-77 CTest skip is not qualification. The GDN/PLE fixture exercises fixed-graph owner/segment changes and accepted-prefix commits. The QSA fixture plans 135 windows, widths 1..8, position 0, page/selection boundaries, 258K and terminal position 262141. Independent fixtures reset Host, pools, maps and indexer state. Follow windows retain accepted state, pre-resolve future append pages and compare resident append before a later resolve could conceal a missing mirror. CLOCK physical slot order is not compared bitwise; logical bijections, selected page bytes, integer counters and attention outputs are checked. These synthetic tests do not replace full-model OFF/ON canonical-state, sampling or parallel-native qualification.

Only then run the frozen 258K workload, unchanged 262144 logical capacity / 20480 resident, one lane and four lanes, at least three paired rounds. Separate warmup by native request generation. Preserve failures and raw integer accounting. Monitor real graph-memory high water and retain the >=300 MiB device-free guard. No throughput or memory-headroom improvement is claimed by this document.
