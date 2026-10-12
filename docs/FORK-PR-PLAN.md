# Independent PR extraction plan

Design IDs are not GitHub PR numbers. Feature branches are prepared locally; no empty feature PR is advertised as delivered. Each dependency-bound task starts only after its predecessor's exact SHA is accepted and recorded. The current starting point is upstream v0.1.43 (`7695859aa13afb248bdb4d36f9b2e95707db079d`).

## Provenance and limits

- Tested donor: `b479ef24f42548bb25c3a03b5a52e8c10b0affd1`.
- Tested v43 fused source: `c15c25f4513ec0d0733cef8c780b1d0538a2753a`.
- Tested fused binary SHA-256: `bfb410bcce1fbb52cde9e77c7490b6e90c9ff3448f47f770b39995aae8edad58`.
- Qualification receipt SHA-256: `34fd42282bc30af9a5cd096978f2e1386224aedba9c765bb84b8ae76a15648c5`.

These are original experiment pins, **not claims that the commits or binaries are published in this fork**. Extraction produces new commits and binaries, which require new tests. The complete donor-to-v43 fusion spans 40 changed files: do not duplicate it in each feature PR.

## Module queue

| Contract | Branch | Start condition | Behavior owned by this PR |
|---|---|---|---|
| [PR-6](pr/06-wide-nonfinite-guard.md) | `pr/06-wide-nonfinite-guard` | Exact v43 pin | Dynamic guard capacity, all-row mask, tests |
| [PR-2a](pr/02a-c2-speculative-foundation.md) | `pr/02a-c2-speculative-foundation` | Accepted PR-6 SHA | Minimal ≤8-row speculative C2/private-slot foundation |
| [PR-1](pr/01-direct-slot-admission.md) | `pr/01-direct-slot-admission` | Accepted PR-2a SHA | Two canonical slots, lazy spill, admission lifecycle/accounting |
| [PR-2b](pr/02b-adaptive12-early-ring.md) | `pr/02b-adaptive12-early-ring` | Accepted PR-1 + PR-6 | spec6/rows12, merged experts and Early-Ring |
| [PR-3](pr/03-narrow-dense-parallel.md) | `pr/03-narrow-dense-parallel` | Accepted PR-2b SHA | ≤8-row dual dense/private streams |
| [PR-4](pr/04-q5q6-wide-head.md) | `pr/04-q5q6-wide-head` | Accepted PR-2b + PR-6 | Native Q5/Q6 wide projection and dispatch observability |
| [PR-5](pr/05-qsa-same-slot-tile.md) | `pr/05-qsa-same-slot-tile` | PR-2b; full coverage on PR-3/4 integration | Same-slot MQ tile selection/attention |
| [PR-7](pr/07-host-ring-diagnostics.md) | `pr/07-host-ring-diagnostics` | Stable runtime interfaces | Optional host/ring accounting and analyzers |

```text
v43 ─── docs/PR-0
 └── PR-6 safety ── PR-2a compatibility ── PR-1 DirectSlot
                                           └── PR-2b adaptive12/EarlyRing
                                                  ├── PR-3 NarrowDense
                                                  └── PR-4 wide head
                                                       └── PR-5 QSA (test with PR-3)
                           PR-7 diagnostics after shared interfaces stabilize
```

PR-2a is compatibility work, not a new throughput percentage. PR-1 owns the admission state and its observability; optional timing diagnostics remain PR-7. Shared `generate.cpp` / `verify.cpp` changes are sequentially rebased, not concurrently overwritten. Adjust boundaries only when a concrete caller/interface dependency is demonstrated and recorded.

## Required contract fields at execution

Every task below must receive an exact executable base SHA, owner and approved test/resource window before code changes. Its current prepared state is **not tested/merged**. The PR body records:

- one measurable goal, exact base/head SHA and dependencies;
- written paths and deliberate non-goals;
- CPU/service/CUDA commands and actual executed counts/exit/log hashes;
- GPU lease, compiler/driver/architecture, bitwise versus numerical comparison semantics;
- fresh C1 regression and C2 paired acceptance, including all negative/EOS/outlier runs;
- public evidence and mapping from original experiment pins to this new PR SHA;
- rollback unit and remaining unjudgeable gates.

See [delivery workflow](FORK-WORKFLOW.md). No prepared branch is merged automatically; no historical performance figure qualifies an untested extraction.
