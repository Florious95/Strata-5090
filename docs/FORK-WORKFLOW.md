# Fork delivery and upstream tracking

## Branch contract

- `origin`: `https://github.com/Florious95/Strata-5090.git`.
- `upstream`: `https://github.com/Niko1221/Strata.git`.
- `upstream-track`: unmodified official history, fast-forward only. Initial pin: v0.1.43 / `7695859aa13afb248bdb4d36f9b2e95707db079d`.
- `main`: reviewed integration target. Every local change, including an upstream sync, uses a PR; no reset to upstream, force-push or direct feature push to main.
- One module, branch and isolated worktree per PR. Dependency branches must be based on an actual accepted predecessor SHA, not an unspecified future branch.

Repository protection is a **required policy, not an assertion that GitHub enforcement is already enabled**. The owner should enable required PR review, the applicable CI checks, and force-push/deletion restrictions once the first workflow exists. Do not auto-merge as part of worker delivery.

## Upstream sync

```sh
git remote add upstream https://github.com/Niko1221/Strata.git # once
git fetch upstream --prune --tags
# Choose a reviewed tag/SHA, not a floating deployment target.
git switch upstream-track
git merge --ff-only v0.1.43^{}
git push origin upstream-track

git switch -c sync/upstream-v0.1.43 origin/main
git merge --no-ff v0.1.43^{}
# Resolve only reviewed module conflicts, build and qualify the exact result.
git push -u origin sync/upstream-v0.1.43
gh pr create --base main --head sync/upstream-v0.1.43
```

The initial main already equals v0.1.43, so this example needs **no sync PR now**. For a later release substitute its pinned tag. Stop if the mirror cannot fast-forward. Keep upstream ancestry in the sync merge; squashing the upstream history into a synthetic replacement loses useful provenance.

Audit each sync for KV layout/residency, slot lifecycle, MTP depth, dense streams, graph metadata, dtype/kernel dispatch, CLI defaults and server cancellation/nonfinite cleanup. Official changes are neither automatically safe nor automatically incompatible.

## Feature PR gate

1. Freeze a [task contract](FORK-PR-PLAN.md): goal, exact base SHA, writer, read/write scope, dependencies, non-goals, failure modes and required tests.
2. Read caller/interface state paths first. Caller, interface and tests change atomically. Do not remove safety guards or change the frozen GGUF assets to make a test pass.
3. Run scoped CPU/service tests and a fresh CUDA build. Pin compiler, target architecture and llama dependency; rebuild all affected ABI components instead of mixing old libraries.
4. Obtain an exclusive GPU lease from the benchmark owner. Test only the approved SHA/window; record source/binary/test/log SHA, fixture counts, executed assertions and exit status. Release all GPU processes afterward.
5. Benchmark in the separate benchmark worktree: frozen manifest, actual258K/full262144Q4, fixed output, reserve, EOS continuation, fresh processes, three paired/interleaved ON/OFF rounds where meaningful. Keep failures, negative pairs and outliers.
6. Test **true solo C1** separately with the same assets and configuration; a C2 process with one idle slot is not a C1 control. Check routed occurrence balance, quality, cache/restore isolation and minimum VRAM headroom.
7. Push the declared branch to origin; open a PR with real SHA/URL and evidence. Independent review examines the exact revision and retained limitations. The worker does not self-merge.
8. After accepted integration, rerun the complete stack at its new SHA. Historical `c15c25f` receipts do not transfer automatically to a cherry-pick or extracted build.

The lightweight documentation CI checks archive integrity, ledger arithmetic and README figures; it is **not** a substitute for compilation, GPU contracts or model benchmarks. Runtime PRs must specify their additional required CI and hardware receipts.

## Resource, publication and release boundaries

- Production and previously qualified experimental trees stay frozen. New work uses isolated clones/worktrees and local configs/ports. Do not restart systemd or expose a benchmark backend as an incidental build step.
- Do not run external/untrusted PR code on the 5090, SSH with production access, or a secret-bearing self-hosted runner. Trusted GPU jobs require manual approval of an exact SHA and an exclusive lease; never cancel a running model workload implicitly.
- Publish safe summaries, hashes, commands and toolchain receipts. Do not publish GGUF files, hidden needle contents, private prompts, API keys or full private configs. Transfer source/artifacts through Git, not SCP.
- Profiler runs are diagnostic, not scoring runs. Unknown clocks, occupancy and hardware traffic stay unknown. CPU↔GPU numerical tolerance must not be called GPU bitwise parity.
- No zero-test, skip, missing fixture or observer failure may be relabelled PASS. An unjudgeable gate remains unjudgeable until measured.
- Merge and production deployment are different authorizations. Pin a qualified release SHA, retain the previous release and receipt, and obtain explicit deployment approval. Roll back a merged feature with a reviewed revert rather than rewriting main history.

## PR extraction order

`guard → C2 foundation → Direct Slot → adaptive12/Early-Ring → Narrow / wide head → QSA`.

Diagnostics are optional and separate. Keyed PLE prefetch, C2 streaming and wider expert tiles are future research PRs, not hidden parts of this release. See [individual contracts](FORK-PR-PLAN.md).
