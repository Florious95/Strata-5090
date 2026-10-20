# Experimental generalized streamed slots and WindowAuditV1

This is an **opt-in, unqualified CUDA experiment**, based on the C2 golden branch at `2938a8441cd1ec785c5222638a0795338fa4c4b0`. No GPU correctness, throughput, hit-rate or expert-capacity result is implied by the CPU tests. The default path stays unchanged.

## Execution contract

Only single-GPU NVIDIA, text-only Qwen3.8-Flash-Next geometry, Q4_0 KV, native experts, MTP, serve and exactly 2/4/8 slots are eligible. Keep each logical context at `--max-context 262144`. Resident KV can be full VRAM or Host-backed `--kv-resident 20480` / `32768`. Grow, pipeline/groups, async adaptation, fixed padded windows, helper/remote experts, other backends and generalized vision are not qualified. Invalid requested combinations fail, rather than silently shrinking the slot count.

- `STRATA_BATCH_GENERALIZED=0|1` (unset = 0): select existing versus generalized execution **in the same binary**.
- `STRATA_BATCH_AUDIT=/absolute/new/file.jsonl` (unset = no writer): minimal buffered audit, exclusive file creation, owner read/write permissions on POSIX; contains no prompt/token content. Both A/B arms must use this mode. Diagnostic `STRATA_TRACE` / host/ring profiling is a separate experiment, not enabled in performance arms.
- `--batch N` comes from the server's `parallel` launcher exactly once. Do not duplicate it in config args. The reserve flag is `--vram-reserve-mib`, not `--reserve-vram`.
- Do **not** pass old `--batch-slot-only` for a C4/C8 A/B: it still explicitly supports only legacy full-resident C2. `INFO slot_only` preserves that CLI meaning; `canonical_slots` reports effective ownership, `batch_generalized` and `batch_window_audit` report native capability activation.
- Existing golden kernel env/flags must be identical between arms, including tail skip **0**, QSA tile, narrow dense parallel, expert merging, wide head and DF_QB_Q4. An env value is not evidence of actual dispatch.

Suggested ON row geometry (requires the existing MTP assets and all baseline model/config paths unchanged):

| slots | `--batch-spec` | `--batch-rows` | proposal rows/slot | full live layout |
|---:|---:|---:|---:|---|
| 2 | 6 | 12 | 5 | 6 + 6 |
| 4 | 3 | 12 | 2 | 3 + 3 / 3 + 3 |
| 8 | 2 | 16 | 1 | 2 + 2 + 2 + 2 / 2 + 2 + 2 + 2 |

Rows include the pending input. Two execution groups are **not** two requests. Physical capacity stays 16 rows with at most 8 columns per dense group. Packing only removes spare proposal rows, never a live slot's pending input. Each slot retains private target, MTP, checkpoint, authoritative Host KV and resident map; slot zero reuses the main target rather than allocating N+1 targets. Generalized QSA tiles follow complete slot segments, independent of dense groups; each streamed tile resolves its own selected-page union before attention. Unsupported union capacity falls back to scalar resolve/attention; map overflow fails before unsafe memory use.

Audit mode prevents server solo `GEN` fallbacks in **both** arms so all output has a native slot/generation owner. Both generalized and audited reference arms also seed an unused owner before reusing only a tiny (<512 token) common header. A meaningful longest-token-prefix, including rewritten 29K reply tails, still wins; if no owner is empty, even a short prefix retains affinity. This is an explicit experimental serving regime, not a claim that OFF + audit is byte-for-byte identical to historical default server scheduling. Remove historical bootstrap overlays when testing this policy, or document an identical overlay on both arms.

### Eight-owner draft overlap

`MtpDrafter::draft_batch` now accepts up to eight distinct private drafter owners. Membership preflight and **all four** host scratch families (limits, counts, two-token PLE history, per-step active mask) share the eight-owner bound. The old four-owner refusal and `[4]` scratch were a missed generalization, not a KV-streaming lifecycle or allocation error. Duplicate/null owners still fail before graph capture; per-owner device/count/accepted-row validation, launch ordering and failure stream synchronization remain. This changes neither per-drafter row/proposal capacity nor verifier16-row capacity, KV residency or graph retention policy. A CPU test covers all256 owner subsets, non-prefix order, duplicates/nulls, nine-owner refusal and private scratch reset/guarding; it does not prove GPU overlap arithmetic or streaming correctness.

### Bounded graph retention and memory diagnostics

Audited or generalized processes default to **4 layout pairs**, versus the existing 64 in ordinary MTP/fast batch serving. Audited scalar batching uses the same limit even without MTP; unrelated unaudited non-MTP batching remains unchanged (INFO limit0 denotes no configured layout cap). Set `STRATA_BATCH_GRAPH_LIMIT=1..64` identically in both arms to override; empty, zero, negative, non-decimal and out-of-range values fail closed. Native INFO reports `batch_graph_limit`. The existing synchronized LRU removes both execute and commit graphs and associated host metadata before capturing a new layout when at capacity. This is a layout-count cap, **not a byte quota or a 300MiB free-memory guarantee**: fixed verifier graphs, capture temporaries and driver caching are separate.

Only for a separate diagnostic arm, `STRATA_BATCH_GRAPH_STATS=1` reports observed device-free bytes after LRU / before capture, after execute capture and after commit capture, plus a signed before/after delta, layout count, limit and rows. It makes no queries or output on replay. These are device-free observations, not pure attributed graph bytes or transient peak measurements. Keep this flag absent in performance arms. More eviction can incur synchronization and recapture cost; quantify that instead of assuming free speedup.

Start the new small-cache safety qualification with `--vram-reserve-mib 2000` and automatic expert sizing in both arms. Preserve every logical 262144 context and chosen resident. Re-measure expert bytes/count/membership, measured integer hit rate, device-free minimum and long-run graph-cache hit ratio; no expert-count or hit-rate guarantee is implied. The observed old 195MiB low-water mark plus an extra600MiB would be only795MiB under otherwise identical conditions, not a proof about future layouts. Do not relax the 300MiB safety guard.

### Independent conversation affinity during the experiment

In a process reporting `INFO batch_generalized=1` or `batch_window_audit=1`, optional HTTP `prompt_cache_key` selects a retained physical owner. Supply a **unique stable string per independent conversation** (1..64 characters), identically in both arms and across warmup, follow-up and EOS requests. The server hashes it internally; it is not emitted in the native audit or inserted into prompt tokens. A new key fills an idle, genuinely empty owner first even when independent prompts share 51K tokens. A known key returns to its retained idle owner; while that key is admitted or reserved, another request with it waits rather than duplicating it on another slot. After all owners fill, normal LCP/LRU eviction remains possible.

Releasing the busy flag does not erase a binding while its KV remains retained. Before each slot lookup, idle empty state invalidates its binding and stale provisional reservation; process restart resets all bindings. Publishing a new retained binding requires successful native admission. The key is a routing hint, not authentication or permission to reuse mismatching KV: native token/image/checkpoint validation remains authoritative. Unkeyed requests retain the LCP policy above. Multiple batch groups are not supported by keyed experimental routing.

This reinterpretation is **only for the opt-in experiment**. In ordinary serving, upstream clients may intentionally share `prompt_cache_key` between parent/subagent requests; the default service does not validate or reinterpret that standard cache-group field as a unique conversation identity. Use independent lane keys explicitly for the experiment. Internal continuation legs retain the same request metadata. C1 without batch slots does not gain a fabricated native owner.

The early generalized option guard rejects only positive explicit `--pipeline-windows`; the default sentinel `-1` resolves to zero for batching later in option resolution. `--pipeline-windows 0` remains a valid explicit setting.

## Wire contract

Each line is one JSON object with required envelope:

```
schema: "strata.window-audit.v1"
run_id: string
kind: "header" | "window" | "admission" | "emit" | "lifecycle" | "footer"
seq: uint64
```

Header is seq 0. Every record, including header/footer, increments seq by one. Footer is last and `records_total = footer.seq + 1`. Independent `window_id` starts at 1. JSON integers are never rounded percentages.

### Header

Required: `slots`, `max_context`, `kv_format:"q4_0"`, `kv_resident_requested`, `kv_resident_effective_by_slot` (length slots, 0 means full resident), `build:{engine_version,source_sha,source_dirty}`, `audit_mode:"minimal"`.

`source_sha` is a 40-hex configure-time Git HEAD, or null when unavailable; `source_dirty` is boolean or null. Fresh configure/build and external source/tree/binary hashes are still required. Never identify an uncommitted binary by a future commit.

`capabilities` has booleans `generalized,canonical_admission,segmented_verify,streaming,audit_all_batch_paths`, integers `max_rows,max_dense_groups` (last = 2).

`provenance` has `counter_schema:"expert-dispatch-occurrences-v1"`, `route_sources` mapping `routed/hits/cpu_entries/offload_entries/skipped` respectively to `ExpertDispatch.experts/cache_hits/multi_entries/offload_entries/tail_skipped`, `snapshot_boundary:"post-reset-pre-target/post-pool-pre-draft"`, `emitted_source:"native-token-protocol-write"`.

### Window

Required: `window_id`, `scope:"target_decode"`, `completed`, `valid`, `invalid_reason` (string/null), `stage_id`, `device_id`, `active_slot_ids`, `scheduled_slot_ids`, `row_slot_ids`, `rows`, `rows_useful`, `rows_padding`, `emitted`, `slots`, route fields and `dispatch`.

`active_slot_ids` snapshots all native decoding-eligible owners before target work. `scheduled_slot_ids` names actual complete segments, in row order, and can be a strict subset. BSTAT queues or an average row count are not substitutes. Each `slots[]` item has `slot_id,request_generation,position_start,rows,useful,keep,emitted,finish_reason`. `emitted` is the count at native token writes, **not** speculative keep. EOS, cancellation and nonfinite handling retain their actual emitted counts.

Route fields: `route_applicable:true`, `route_counter_valid`, `route_scope:"target_computed_rows"`, and all five independent nonnegative int64 counts `routed,hits,cpu_entries,offload_entries,skipped`. They are target-computed occurrences before commit, including rejected proposals, not deduplicated expert jobs or useful output. Require:

```
routed > 0
routed = hits + cpu_entries + offload_entries + skipped
```

Snapshot after reset/before target and after joined target pool/before MTP draft. Invalid snapshots yield null counts and invalid records, never synthetic zeros or a hits-derived denominator.

`dispatch` required keys: `canonical_admission,generalized,merged,dense_groups,dense_parallel,shared_after_ring,head_type,head_wide_active,qsa_tile_calls,qsa_tile_rows,qsa_scalar_rows,kv_mode,kv_resolve_strategy,kv_resolve_calls,graph_cache_hit,fallback_reason`. The resolver map is exactly `{"scalar_per_row": integer>=0, "union_per_slot": integer>=0}`: **layer-summed actual call counts**, not queries/pages. Strategy is `none` for 0/0, `scalar_per_row` or `union_per_slot` when only that count is positive, `mixed` when both are positive. QSA calls/rows are also layer-summed actual dispatch metadata, restored on graph replay.

### Admission, emit and lifecycle

Admission required: `admission_id,scope,slot_id,request_generation,has_target_run,completed,valid,invalid_reason,emitted,canonical_admission`, route fields, `copy_counter_valid`, and nullable `main_to_slot_kv_bytes,slot_to_main_kv_bytes,park_bytes,restore_bytes`.

- Target run: scope `admission_target`, applicable true, normal target route fields.
- No target: scope `admission_no_target`, applicable false, **route_counter_valid/route_scope/all five route counts null**. This legitimate N/A is not zero and need not be an invalid record.
- Four byte fields measure **encoded logical historical KV + pooled-index payload per successful operation**. Main/slot copy counts that payload once, not separate read and write bus traffic. Park counts the successfully retained snapshot's KV payload (including reused blocks); restore counts a successful full image or restored prefix KV payload. They exclude checkpoint/GDN/PLE metadata, vector capacity and stream-page DMA, and are **not PCIe traffic measurements**. Missing instrumentation makes all four null with `copy_counter_valid:false`. A canonical zero proves no main-slot historical KV copy, not zero Host-to-GPU streaming.
- Admission error scopes are drained with `completed:false,valid:false`; their campaign cannot qualify. No successful footer may erase failed work.

Reserved `emit` records describe output not already counted in a window/admission: required `scope:"non_target_output",slot_id,request_generation,emitted,route_applicable:false`. The current audited native paths attribute all tokens to admission/window and emit zero such records; do not create balancing emits to conceal missing target work.

Lifecycle required: `slot_id,request_generation,event,reason` (nullable). Generation starts at 1 and increases per physical slot, never inferred from a server reservation. State flow `claim -> admitted -> done -> drained`; an error/cancel before admission can go `claim -> done -> drained`. Reuse records `reused` with the **new** generation before its claim; the previous generation must have drained.

### Footer and HTTP boundaries

Footer required: `records_total`, `kind_counts` for all six kinds including zeros, `invalid_records,dropped_records`, `scope_totals`, `slot_generation_totals`.

Each scope total has `scope,records,route_records,route_applicable,route_counter_valid,routed,hits,cpu_entries,offload_entries,skipped,emitted`. No-target route values stay null; invalid-route totals remain invalid rather than silently dropping bad windows. Each slot/generation total has `slot_id,request_generation,window_emitted,admission_emitted,other_emitted,emitted`, with exact conservation. Footer is whole-process accounting, **not** the measured-request hit rate.

Audited native `DONE` appends `slot_id=N request_generation=G`. HTTP final `timings` carries these plus an ordered, adjacent-deduplicated `slot_generations:[{slot_id,request_generation},...]` for **every** native continuation/BYIELD leg. The top-level pair is only the last leg. Chat/Anthropic/Responses preserve the identity; without a valid native pair the server does not invent one or change legacy response shapes. Filter warmup versus measured windows using complete HTTP leg identities, not timestamps, process footer percentages or server-local counters.

## Additive Solo-in-slot diagnostics (experimental Commit A)

`STRATA_BATCH_SLOT_SOLO=1` is a separate default-off C4 experiment; see [C4-SLOT-SOLO.md](C4-SLOT-SOLO.md) for limits and graph ownership. Existing V1 event kinds, sequence rules, scopes, integer fields, HTTP identities and lifecycle are unchanged.

Header `capabilities` additionally exports `slot_solo:bool`, `row_table:false`, `slot_solo_target_graph_limit:int` (0 OFF; 1..8 ON, configured default 4), and `slot_solo_commit_graph_limit:int` (0 OFF; 4 ON). These bound executable counts, not MiB.

Every native target window adds `dispatch.execution_path:string` (`legacy|slot_solo`), `graph_key_kind:string` (`legacy_layout|slot_solo`), `graph_key_rows:int` (actual rows), `graph_owner_slot:int` (Solo native owner; legacy -1), and `graph_address_epoch:uint64` (Solo owner-address/static-profile epoch >=1; legacy 0). A request generation does not change the address epoch. This is **not** a rows-only graph ABI. On Solo, all active/scheduled/row owners must be that owner, capacity stays four slots/12 scratch rows, and actual target width is <=8. Batched optimization flags are not set by capability alone; Solo reports its own captured head/attention/resolve metadata.

## Required physical qualification

Fresh CUDA 13.3 / sm_120 build, binary/source/config identities; exact frozen rendered prompt IDs; scalar-reference versus union attention/commit parity; C2 regression; C4/C8 active owner/row/resolver evidence; exact whole-HTTP route/emitted reconciliation; fail-closed invalid counters; EOS/cancel/nonfinite and N+1 RAM park/restore; memory/pinned-RAM/OOM and graph recapture/reuse bounds. Only then run alternating fresh-process OFF/ON cohorts with identical assets, contexts, generation tokens, clocks and minimal audit. Keep slow/failed/EOS rounds, all raw logs and unrounded integers. No CPU result establishes GPU safety or a throughput uplift.
