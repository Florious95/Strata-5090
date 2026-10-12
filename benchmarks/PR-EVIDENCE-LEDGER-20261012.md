# PR/README evidence ledger

Authoritative raw receipt hashes: `results/PR-EVIDENCE-LEDGER-20261012.json`.

**Chronology, not additive gains.** IQ3_XXS; actual258K/full262144Q4. Hot emitted HTTP output throughput. Cold/warm phases excluded.
Table minima cover ON/observation cells; OFF-control minima and complete run lists are in the JSON ledger.

| Stage | Source | Pooled / peak tok/s | Same-build control / gain | Minimum hit / free MiB | Qualification scope |
|---|---|---:|---:|---:|---|
| C1 historical serial queued4 | `ba5137e` | 165.31 / 169.42 | — | 95.349% / 899 | Historical C1/spec4; 4x2048 outputs, one active slot. Not a current official43 C1 distribution. |
| C2 historical 8-row baseline | `ba5137e` | 172.89 / 175.63 | — | 94.328% / 693 | Two active slots, CLI spec4/rows8; strict 4x2048 queued4. |
| Direct Slot | `d8b0d9d` | 168.96 / 168.96 | — | 95.731% legacy proxy / 293 | One dual2048 run, not queued4 or a paired ablation. 293MiB free fails present >=300MiB rule; legacy hit proxy only. Observer SHA was not recorded. |
| 12-row Dense + Early-Ring bundle | `ba5137e` | 177.96 / 190.80 | 172.89 / +2.93% | 94.673% / 395 | CLI spec6/rows12 versus spec4/rows8 bundle, NOT Early-Ring alone; gains +12.07/+0.503/-2.862%, unstable. |
| Narrow Dense parallel | `300462d` | 189.77 / 193.93 | 183.84 / +3.23% | 95.165% / 331 | Same binary ON/OFF; +2.834/+1.050/+5.794% pairs, not three significant wins. |
| Actual Q5_K wide LM Head | `4e57d4c` | 196.33 / 205.61 | 191.38 / +2.58% | 95.072% / 327 | Same binary ON/OFF; +10.067/+2.748/-4.335% pairs. Preserve third-ON682ms verify spike; no outlier trimming. |
| QSA Tile | `b479ef2` | 207.78 / 214.59 | 191.83 / +8.32% | 94.532% / 329 | Same binary ON/OFF +1.3075/+17.0989/+7.0867%; EOS/routes/hit differ and R1 modest. Not three significant wins or isolated identical-workload kernel timing. |
| v0.1.43 + golden C2 fusion | `c15c25f` | 209.99 / 213.60 | — | 95.084% / 343 | Three fresh all-ON repeats, not paired against old version;209.99 is inheritance evidence, not causal1.06% upgrade gain. |

## Mechanism evidence / attribution limits

### C1 historical serial queued4
No C2 optimization attribution.
Raw archive Git: `13d26c8e6b46dd53c6cb818f812a8f1ca2644f4b`.
Summary: `ba5137e-queued4-matrix-summary.json`.

### C2 historical 8-row baseline
Baseline for the contemporaneous 12-row bundle.
Raw archive Git: `13d26c8e6b46dd53c6cb818f812a8f1ca2644f4b`.
Summary: `ba5137e-queued4-matrix-summary.json`.

### Direct Slot
Removed third solo target/draft allocation and first GEN1->slot full KV copy; code/audit fact, not an isolated measured speedup.
Raw archive Git: `6cc946934e2232e9c3fdab527fcae1c7645ff7b6`.
Summary: `long-d8b0d9d-slotonly-dual-r2.json`.

### 12-row Dense + Early-Ring bundle
Actual rows>8 merged graph/Dense parallel and early ring dispatch; <=8 already had Doorbell overlap.
Raw archive Git: `13d26c8e6b46dd53c6cb818f812a8f1ca2644f4b`.
Summary: `ba5137e-queued4-matrix-summary.json`.

### Narrow Dense parallel
<=8 dual native dense_groups/dense_parallel/shared_after_ring=2/1/1 versus OFF1/0/0; wait reduction8.44–11.26%, host-visible sequence wall not pure GPU time.
Raw archive Git: `81cbe722ef0f6f6217ce5d4808b2f1046c0730c2`.
Summary: `300462d-narrow-queued4-matrix-summary.json`.

### Actual Q5_K wide LM Head
Actual head_type13/head_wide_active1 for9..12rows; avoid second logical437043200-byte weight payload pass. Not a hardware HBM byte measurement; earlier Q6-only candidate was a NO-OP on this Q5 asset.
Raw archive Git: `cb0d56f10e06b7200e27eb50498c8400a9d2d215`.
Summary: `4e57d4c-khead-queued4-matrix-summary.json`.

### QSA Tile
94,212 tile calls /96.0523% tile-row coverage across3ON runs;1,132,080 logical captured selection/attention nodes removed, not hardware memory traffic.
Raw archive Git: `94a0f104fddc2db1d9be8cac6b7a8f7f5b208c04`.
Summary: `b479ef2-qsa-queued4-matrix-summary.json`.

### v0.1.43 + golden C2 fusion
Official43 nonfinite guard expanded to dynamic max_t with3618GPU canary cases; NaN guard ON, no model-run events. Rejected native geometry test remains archived; final corrected qualification exit0.
Raw archive Git: `af78fab950cf6ece7200148aa79ab31cc4fac57e`.
Summary: `c15c25f-fusion43-queued4-summary.json`.

## Reviewer guardrails
- pooled=sum emitted HTTP tokens / sum complete HTTP wall; do not arithmetic-average run rates
- show peak separately; do not claim community instantaneous Peak equivalence
- stage-to-stage deltas change builds/protocol/spec/reserve and are not additive causal gains
- single Direct Slot legacy point is historical, not present qualification
- C2 means <=2 active decoders with four queued requests, never C4
- retain failures/EOS/spikes/negative controls and include minimum hit/free, not best-run only
- profiler-only runs excluded from official throughput; modeled bytes/counts are not hardware bandwidth counters
- publish real fork PR/commit URLs only after commits/receipts are pushed; do not invent URLs

Historical memory-clock/OC state is unknown. New NVML observations must not be retroactively assigned to old runs.
This ledger binds original source commits; PR/cherry-pick commits must be separately mapped once the fork PRs exist.
