# Strata-5090

**A PR-driven RTX 5090 research fork of [Strata](https://github.com/Niko1221/Strata), focused on long-context, multi-agent inference.**

本仓库追踪官方上游，将单卡 5090 的长上下文双路优化拆成可审查、可回退的独立 PR；先测试，后发布。

> **Integration status:** this README documents a previously tested experimental stack. The feature PRs below are being extracted; they are **not yet merged into this fork's `main`**. Benchmark numbers belong to the frozen source/binary hashes in the evidence ledger, not to arbitrary builds of this repository. Production deployment requires separate approval.

## Scope

- Target cohort: RTX 5090 D / Ryzen 9 9950X3D / 96 GB RAM, Linux, Qwen3.8-Flash-Next GSQ-RCO **IQ3_XXS**, actual **258,000-token** prompts, full **262,144-cell Q4_0 KV** per active slot.
- **C2 = at most two active decoders**, with four queued requests, each emitting 2,048 HTTP tokens. It does not mean four active GPU sessions, nor twice the single-stream throughput.
- The tested stack retains upstream v0.1.43 behavior and its NaN/Inf guard. True C1 is a separate regression gate, not a C2 run with one idle slot.
- C2 KV streaming / `--kv-resident 32768`, wider expert GEMV tiles and 400 tok/s are **research proposals, not implemented or measured claims**. Rejected keyed PLE prefetch is not part of the accepted stack.

## Independent PR modules

The IDs below are **design IDs, not GitHub PR numbers**. [Task contracts and dependencies](docs/FORK-PR-PLAN.md) define the extraction order; links point to plans until real PRs exist.

| Module | Responsibility / physical mechanism | Dependency | Delivery status |
|---|---|---|---|
| PR-0 · Repository governance ([GitHub #1](https://github.com/Florious95/Strata-5090/pull/1)) | Archive upstream README, evidence-bound documentation, upstream sync and PR gates | v0.1.43 | Open; [workflow](docs/FORK-WORKFLOW.md) |
| [PR-6](docs/pr/06-wide-nonfinite-guard.md) · Wide NaN/Inf guard | Allocate/consume guards by verifier capacity, including 9–12 rows; keep checks enabled | v0.1.43 | Prepared; safety prerequisite |
| [PR-2a](docs/pr/02a-c2-speculative-foundation.md) · C2 compatibility foundation | Slot-private drafts and segmented accepted-prefix state; preserve v43 lifecycle | PR-6 | Prepared; no standalone speed claim |
| [PR-1](docs/pr/01-direct-slot-admission.md) · Direct Slot Admission | Remove third solo target/draft state and full admission KV copy; lazy RAM spill | PR-2a | Prepared; allocation/copy mechanism proven |
| [PR-2b](docs/pr/02b-adaptive12-early-ring.md) · Adaptive 12-row + Early-Ring | Per-slot spec6, merged rows12, earlier expert dispatch and overlap | PR-1 + PR-6 | Prepared; measured as a bundle |
| [PR-3](docs/pr/03-narrow-dense-parallel.md) · Narrow Dense | Preserve two-stream/private-state parallelism in dual-slot windows ≤8 rows | PR-2b | Prepared; same-binary ablation |
| [PR-4](docs/pr/04-q5q6-wide-head.md) · Wide LM Head | Q5_K/Q6_K 9–12-column projection with one logical weight pass and unchanged per-column arithmetic | PR-2b + PR-6 | Prepared; actual model head is Q5_K |
| [PR-5](docs/pr/05-qsa-same-slot-tile.md) · QSA Tile | Aggregate same-slot scores → topk → attention with MQ≤8; safe scalar fallback | PR-2b; PR-3 for narrow coverage, PR-4 metadata | Prepared; same-binary ablation |
| [PR-7](docs/pr/07-host-ring-diagnostics.md) · Optional diagnostics | Integer-nanosecond host/ring accounting; disabled for benchmark scoring | Stable module interfaces | Prepared; not a speed feature |

Caller, interface and contract tests stay in the same PR. The speculative foundation is necessary compatibility work: copying the entire fused patch into six overlapping PRs would not produce independent features.

## Aggregate benchmark evolution

**Chronology, not additive gains.** Pooled throughput = total emitted HTTP tokens / total complete HTTP wall time. Cold load and initial warm-up are excluded; request admission, queueing and continuation remain in the hot HTTP wall. Peak is the best complete run, not an instantaneous decode burst.

| Stack / aggregation layer | Frozen source | Pooled tok/s | Peak run tok/s | Minimum hit / free MiB | Evidence boundary |
|---|---|---:|---:|---:|---|
| Historical C1 · serial queued4, spec4 | `ba5137e` | **165.31** | 169.42 | 95.349% / 899 | Three runs; not current official43 C1 distribution |
| ↳ Historical C2 · 8-row baseline | `ba5137e` | **172.89** | 175.63 | 94.328% / 693 | Two active decoders; spec4/rows8 |
| &nbsp;&nbsp;↳ 12-row Dense + Early-Ring bundle | `ba5137e` | **177.96** | 190.80 | 94.673% / 395 | spec6/rows12 bundle; not Early-Ring alone |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ Narrow Dense parallel | `300462d` | **189.77** | 193.93 | 95.165% / 331 | Same-binary ON/OFF |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ Q5_K wide head | `4e57d4c` | **196.33** | 205.61 | 95.072% / 327 | Same-binary ON/OFF; negative third pair retained |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ QSA Tile | `b479ef2` | **207.78** | 214.59 | 94.532% / 329 | Same-binary ON/OFF; routes/EOS vary |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ v0.1.43 + complete golden stack + wide guard | `c15c25f` | **209.99** | 213.60 | 95.084% / 343 | Three fresh all-ON runs; not an isolated v43 gain |

The chronological C1-to-final difference is **+27.03%**, not an isolated C2 concurrency gain. Stages differ in builds, speculation and protocols; do not add or multiply their deltas.

### What the controlled comparisons establish

Each row compares the **same binary**; the control is contemporaneous, not necessarily the preceding chronological stage. Each arm contains 24,576 emitted tokens across three fresh-process runs.

| Feature / bundle | ON pooled | Control pooled | Pooled change | Important limitation |
|---|---:|---:|---:|---|
| 12-row Dense + Early-Ring | 177.96 | 172.89 | **+2.93%** | Changes spec depth and row width together; one negative pair |
| Narrow Dense | 189.77 | 183.84 | **+3.23%** | Positive pairs, not three independently significant wins |
| Q5_K wide head | 196.33 | 191.38 | **+2.58%** | Third pair −4.335%; 682 ms verify spike retained |
| QSA Tile | 207.78 | 191.83 | **+8.32%** | EOS/routes/cache hits differ; not identical-workload kernel timing |

**Direct Slot is a shared structural prerequisite, not another percentage.** Its historical 168.96 tok/s point was one dual-2048 observation, used a legacy hit proxy and had only 293 MiB free—below today's 300 MiB gate. It is not included as a qualified rung or an isolated throughput gain.

### Official v0.1.43 reference: 139.79 → 209.99

| Reference | HTTP tok/s | Qualification |
|---|---:|---|
| Pure official43 C2, one queued4 observation | **139.79** | Observer exit 1 retained; exact C2 routed/gap instrumentation missing and rounded native hit <93%; **not performance-qualified** |
| v43 + golden stack, three all-ON runs | **209.99** | Hash-bound aggregate; minimum true hit 95.084%, free 343 MiB |

This is an **observed +50.22% difference**, not a paired, single-variable or production-qualified gain. Official43 also had one true-C1 HTTP observation of **185.10 tok/s**, not a C1 distribution. Official C2 kept the solo state plus two target slots and performed six full admission copies totalling **2,652.1 ms**. Its 5,010 MiB expert-cache reduction relative to C1 is not all recoverable by Direct Slot alone. See the [official observation receipt](docs/evidence/official43-observation.json).

## Evidence and safety

- [Readable ledger](benchmarks/PR-EVIDENCE-LEDGER-20261012.md) · [machine-readable ledger](benchmarks/results/PR-EVIDENCE-LEDGER-20261012.json) · [local hash-validation receipt](docs/evidence/ledger-import-verification.json).
- SHA-256 binds the original receipts; it is an integrity hash, **not a digital signature**. Raw model assets, prompts, hidden needles, keys and private configuration contents are not published. Raw receipt hashes are recorded for audit; the ledger alone does not make every original raw file publicly available.
- Actual LM head: **Q5_K, type 13, 437,043,200 bytes**. Wide dispatch records `head_type` and `head_wide_active`; a flag alone is not proof of execution. The superseded Q6-only candidate was a no-op on this asset.
- QSA ON evidence: **94,212 tile calls**, **96.0523% tile-row coverage**, **1,132,080 logical graph nodes removed**. These are dispatch counts, not measured GDDR traffic.
- Original fused qualification: 18 CPU contracts, 362 server tests, 14 trace-analyzer tests; Q5/Q6 wide-head bit parity, 44 QSA fixtures / 132 dual-stream replays, 72 GDN layouts / 882 partial commits, and **3,618 NaN/Inf GPU canary cases**. A rejected native geometry case remains recorded; corrected qualification exited 0. These receipts **do not qualify new extraction commits**.
- Historical memory OC state is unknown. User-reported 32 Gbps memory implies nominal 2,048 GB/s versus stock 1,792 GB/s, but does not establish measured bandwidth utilization or historical clocks. Power/TDP is not “remaining performance percentage.”

## Tracking upstream and contributing

`origin` points to this fork; `upstream` points to official Strata. `upstream-track` is an unmodified fast-forward mirror; `main` is the PR-only integration target. Sync upstream through a reviewed PR, never by resetting the integrated fork.

See [workflow and release boundaries](docs/FORK-WORKFLOW.md), [module contracts](docs/FORK-PR-PLAN.md), and the [PR template](.github/pull_request_template.md). Every runtime PR needs a frozen base, independent review, scoped CPU/CUDA contracts, an approved GPU lease and fresh C1/C2 acceptance. A new source SHA requires new receipts. Missing or skipped tests are not PASS.

For upstream setup and general model support, use the [archived original v0.1.43 README](docs/archive/README.upstream-v0.1.43.md) and [upstream docs](https://github.com/Niko1221/Strata/tree/7695859aa13afb248bdb4d36f9b2e95707db079d/docs). Do not deploy a floating branch on the strength of historical benchmark numbers.

## Attribution

Built on [Niko1221/Strata](https://github.com/Niko1221/Strata), under the unchanged [MIT License](LICENSE). [Archive provenance](docs/archive/README.md) preserves the original README byte-for-byte and explains its relative-link context. Upstream authors, dependencies and model licenses remain credited in [HOW_IT_WORKS](docs/HOW_IT_WORKS.md#credits).
