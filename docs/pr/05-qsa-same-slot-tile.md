# PR-5 · Same-slot QSA multi-query tile

- Branch: `pr/05-qsa-same-slot-tile`.
- Execution base: accepted PR-2b; full narrow-coverage qualification on PR-3 + PR-4 metadata integration. Exact SHA currently pending.
- Goal: aggregate same-slot consecutive rows into MQ≤8 scores/topk/attention dispatch so pooled index blocks serve the tile rather than one query launch at a time.
- Scope: QSA invocation/guards in `verify.hpp/.cpp`, native dispatch metadata and consumers, `tests/core/qsa_tile_*`, QSA trace analyzer/tests and scoped CMake registration. Reuse existing multi-query kernels; do not change their arithmetic.
- Required invariants: `batch_rec_`, GDN and KV append order, causal query boundaries, distinct slot state and two-stream Merged/Narrow topology remain unchanged. Only full-Q4 / kv_mode0 / valid same-slot layouts enter the fast path; all unsupported edges take scalar fallback.
- Non-goals: C2 streaming, moving append order, combining different slots' KV, changing selected blocks/topk semantics or breaking narrow parallelism.
- Contracts: CPU tile guards and exhaustive boundary layouts; GPU bit parity versus scalar scores/topk/attention, two streams with poison and captured replay. Historical qualification had44 fixtures/132 dual-stream replays; new SHA needs new receipts.
- Observability: native tile/scalar calls, rows and coverage; analyzer checks dispatch conservation. Historical three ON runs recorded94212 tile calls,96.0523% tile-row coverage and1132080 logical graph nodes removed—not measured memory traffic.
- Historical evidence:207.78 versus191.83 pooled (+8.32%), pairs +1.3075/+17.0989/+7.0867%. EOS/routes/hits differ; not identical-workload kernel timing or three significant wins.
- New acceptance: true C1 regression, frozen C2 paired matrix, full narrow/merged coverage and all fallback paths. Keep all EOS/negative/outlier evidence.
- Rollback: tile dispatch selector with scalar fallback; do not remove other merged/dense improvements.

Freeze integration SHA and resource lease before implementation; submit actual test/log/binary hashes. See [workflow](../FORK-WORKFLOW.md).
