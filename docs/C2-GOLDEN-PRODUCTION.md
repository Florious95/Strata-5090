# Authorized v43 golden C2 production cutover

This is an operator runbook, not a claim that installation has already happened. The user explicitly authorized production and LAN/Tailscale access. `bench_dev` owns the target GPU, cutover and acceptance; the delivery worker does not start another backend.

## Release and network pins

- Host: `5090-omarchy`, account `florious`.
- Published runtime: `/home/florious/Projects/Strata-c2-v0.1.43-port`, source `77cd2b2de043721bda8deff2fd691e88d7499a2a` ([runtime PR#3](https://github.com/Florious95/Strata-5090/pull/3)).
- Engine: `build/strata`, SHA-256 `640c5a6600e687c23c870a99d16b3c59bd769c86aefd0b331460371fa1c950d8`.
- Independently built staging artifact: `/home/florious/Projects/Strata-c2-vision-direct-research-dev-20261012/build/strata`. Publish only after the owner stops the production control group; never overwrite a live executable.
- Matching `serve/server.py`: SHA-256 `06ca08c2497f4200a46d602b88e8b87df91e79b8fc2af0caf855b4da7358ddb1`.
- Assets: previously qualified **IQ3_XXS**, full262144 Q4_0 KV, C2 spec6/rows12, reserve1400. Do not substitute the old production IQ3_S/INT8/32K-streaming config or a new GGUF.
- Target confirmed by benchmark owner: GPU2MiB/no compute process, port8080 free, Tailscale IPv4 **100.99.91.50**.
- Listener: **0.0.0.0:8080**. LAN `http://192.168.31.110:8080`; Tailscale `http://100.99.91.50:8080`; optional existing MagicDNS `5090-omarchy.tailbfe94c.ts.net`.
- These are LAN/tailnet endpoints, not Internet publication. Do not enable Tailscale Funnel, router port-forwarding, permissive global CORS or a public reverse proxy. Use TS for encrypted transport; LAN HTTP requires a trusted LAN.

## Files and exact production config treatment

- New independent user unit: `~/.config/systemd/user/strata-c2-golden.service`; source [ops/strata-c2-golden.service](../ops/strata-c2-golden.service).
- New config: `~/.config/strata-c2-golden/production.json`, mode600.
- Config SHA binding: `~/.config/strata-c2-golden/release.env`, mode600, contains only `STRATA_PRODUCTION_CONFIG_SHA256`, **no secret**.
- Private engine log: `~/.local/state/strata-c2-golden/engine.log`; journal contains service diagnostics.
- Existing key source: `/home/florious/Projects/Strata/strata-iq3_s.json`. It remains untouched. The launcher reads the key **on target into process memory**, supplies `STRATA_API_KEY` to the server, and never puts it in argv, new config, output or Git.

The production config is derived from the **exact frozen golden test config and its SHA**, not recreated from a partial README command. Asset args, asset `cwd`, tokenizer, lib directories and model/sampling parameters remain unchanged. Production-only replacements:

```json
{
  "exe": "/home/florious/Projects/Strata-c2-v0.1.43-port/build/strata",
  "host": "0.0.0.0",
  "port": 8080,
  "parallel": 2,
  "lazy_load": false,
  "idle_unload_s": 0,
  "open_browser": false,
  "api_monitor": false,
  "cors_origins": [],
  "trusted_origins": [],
  "log": "/home/florious/.local/state/strata-c2-golden/engine.log",
  "env": {
    "STRATA_BATCH_DENSE_PARALLEL": "1",
    "STRATA_BATCH_NARROW_PARALLEL": "1",
    "STRATA_BATCH_SHARED_AFTER_RING": "1",
    "STRATA_BATCH_HEAD_WIDE": "1",
    "STRATA_BATCH_QSA_TILE": "1",
    "STRATA_TOKEN_GAP_RAW": "1",
    "STRATA_PLE_KEYED_PREFETCH": "0",
    "STRATA_NAN_GUARD": "1",
    "STRATA_ROUTE_TAIL_SKIP": "0"
  }
}
```

This is the **overlay**, not a standalone runnable config: the generator keeps the complete qualified input and adds release metadata/key-source/Python paths. It rejects changed capacity/dtype/reserve/PCIe/pool/spec/approximation, missing assets, dirty candidate source or mismatched engine/server SHA. Profiles/device-plan/injection must be absent, not set to0. It preserves the qualified pipeline, rather than rebuilding an approximate config from these visible settings.

## Install on target (Git only)

Obtain the benchmark owner's absolute frozen config path/hash and the working Python venv path. Do not send its content/key to another machine. Verify those inputs before running:

```sh
# On5090, as florious; a new delivery clone, never the production/candidate tree.
git clone https://github.com/Florious95/Strata-5090.git ~/Projects/Strata-5090-delivery
cd ~/Projects/Strata-5090-delivery
git fetch origin ops/c2-golden-production-20261012
# REVIEWED_SHA is the exact 40-character delivery SHA handed off by Leader/PR review.
git checkout --detach "$REVIEWED_SHA"

bash ops/deploy-c2-golden.sh \
  /absolute/path/to/accepted-golden.json \
  THE_ACCEPTED_CONFIG_SHA256 \
  /absolute/path/to/working-venv/bin/python
```

The default official venv may be `/home/florious/Projects/Strata/.venv/bin/python`; **verify, do not assume**. The unit launcher is `/usr/bin/python3`, but it execs the supplied qualified venv Python for `-m serve.server` from the matching candidate tree. The delivery checkout must be at the specified stable path; checkout an independently reviewed SHA, not a moving branch.

The script prepares and validates before stopping anything, stops/disables only the owned old `strata-iq3.service`, checks the GPU/port again, installs/enables the new unit and enables user lingering for reboot persistence. Unknown GPU processes are not killed. Restart is always/10s with control-group cleanup and45s stop timeout. Mount/source/config/auth failures refuse launch; the operator must inspect repeated preflight failures rather than bypass them.

It does **not** mount a missing model filesystem, change GPU clocks, alter firewall/TS ACLs, reset GPU, re-download weights or merge either GitHub PR. If one prerequisite is unavailable, fix that owned prerequisite explicitly. Existing config files are not overwritten on rerun; back up the new config and release.env deliberately before regenerating.

## CPU vision release and controlled update

The original c15 text release cannot accept golden slot-only images. Runtime PR#3 now implements direct BGENI, per-slot image banks and saved rotary deltas while retaining existing Dense/Narrow/QSA dispatch. Its **new Linux Release** passed17 scoped CPU tests and hardware contracts in0.927s:1512 bitwise RoPE fixtures/4536 captured replays,3985 snapshot checks; GPU returned to0 compute PID/2MiB. Receipt log SHA `ad0a5c2c0d4ca7b36d5a816a2e362f6095f6f73e3068787238ddc6fbf2ca8343` in the staging tree's `build/vision-gpu.log`. This is not real image identification or new258K performance evidence.

For the benchmark owner's approved CPU encoder, add the complete `vision` object to a **target-only derived input config** before regeneration: `exe`, `mmproj`, `model`, `gpu: false`, and approved positive integer `threads`/image token caps if set. The text model remains IQ3_XXS; the encoder's `model` is vocabulary metadata, not a second GPU model. The launcher checks assets/executable/GGUF headers and refuses GPU vision, misplaced text-engine `--mmproj`, or invented `vision_on_demand`. No secret is moved into this input or output.

Controlled cutover order (bench owner only):
1. Stop `strata-c2-golden.service`, confirm its cgroup/GPU releases. Preserve the c15 source/binary/config and old ops SHA62d18827 for rollback.
2. In the published runtime tree, require tracked source clean, Git-fetch runtime PR#3 and checkout exact77cd2b2. Do not reset unknown changes.
3. Install the independently built binary under a temporary filename in `build/`, verify its full640c5a66 SHA, then atomically rename to `build/strata` while stopped. Source/binary/server now match the new pins. The build uses its own Git-pinned GGML source/fresh ABI; no old object library was reused.
4. Git-checkout the exact reviewed **new ops release SHA** in the delivery clone. Back up existing production.json/release.env privately, then regenerate from the approved complete CPU-vision input and its SHA using the existing deploy script (or its prepare/check commands). Do not merely edit production.json while leaving `_production` or release.env stale.
5. Enable/start the existing independent C2 unit; run local and TS authentication/generation checks, then actual known-image identification in both slots and mixed text/image. Record native `slot_vision=1` and `slot=N images=M rope_delta=D direct=1 solo_kv=0` receipts.

CPU vision does not reserve a GPU encoder; keep reserve1400 initially but measure first/max/multi-image minfree and CPU contention. Two stable image banks cost about6MiB total. There is no standalone on-demand unload. Picture hot-slot reuse remains conservative; RAM offsets round-trip, but nonzero-offset SAVE refuses the old disk schema instead of dropping metadata. Full long-text regression remains separately bench-owned. Rollback must restore **all** source/binary/config/ops pins together, not just one file.

## Acceptance (must return actual receipts)

1. Local script smoke: loaded health200, unauthenticated models401, authenticated models200, two concurrent short chat requests200/nonempty. API keys stay in target process memory.
2. The script repeats authenticated two-request generation through `http://100.99.91.50:8080`. Record its actual result; API concurrency alone is not proof of two GPU-active slots.
3. From the real controller, with no key copying, verify health and unauthenticated rejection:
   ```sh
   curl --noproxy '*' -fsS http://192.168.31.110:8080/health
   curl --noproxy '*' -fsS http://100.99.91.50:8080/health
   curl --noproxy '*' -o /dev/null -w '%{http_code}\n' http://100.99.91.50:8080/v1/models
   ```
   Health must show loaded=true/api_key=true; unauthenticated API must be401. Real TS client generation uses its already provisioned key—never post it to the team.
4. Confirm `systemctl --user is-enabled/is-active strata-c2-golden.service`, `loginctl ... Linger=yes`, old unit disabled/inactive, and listening socket0.0.0.0:8080.
5. Exactly one target engine compute PID; `/proc/PID/exe` is the pinned candidate engine and its cgroup belongs to `strata-c2-golden.service`. No unknown desktop/model process. Record engine/server/config/script/unit SHA and venv version, not secrets.
6. Native INFO must show engine0.1.43, batch_slots2/spec6/rows12, slot_only1/lazyspill1/adaptive_rows1, fullQ4/kv_resident0; verify real dual-active state, narrow2-stream dispatch, head_type13/wide-active in9..12 windows and QSA tile accounting on an approved workload. Short API smoke cannot prove full258K dispatch/head coverage; use the existing qualification plus deployment-native receipts, not invented fields.
7. No actual CUDA error/Xid/OOM/nonfinite event; do not fail simply because startup says “NaN guard ON”. Minimum free≥300MiB/true routed hit≥93% on the appropriate long acceptance workload. Retain all failures/trajectory changes and don't claim209.99 is newly measured production speed.
8. Check the existing host firewall permits trusted LAN and tailscale0 only as needed, TS backend Running and no Funnel/public forwarding. Do not flush/replace firewall rules; add narrow rules through the existing backend only if connectivity is blocked and owner approves.

If the local smoke fails, the installer disables/stops the failed new unit and exits nonzero; old files are preserved. A later TS/controller/native gate failure is **not** full deployment PASS—operator must report and resolve or explicitly roll back.

## Explicit rollback (old IQ3_S C1, not C2)

```sh
systemctl --user disable --now strata-c2-golden.service
# Wait until its control group is gone; investigate any remaining/unknown GPU PID.
systemctl --user enable --now strata-iq3.service
```

This restores the prior v42 IQ3_S/INT8/32K-streaming C1 service, not the new C2 configuration. Old source/config/binary remain unchanged. Do not run the old recovery skill/script while C2 is live: it would restart the competing legacy service. Benchmarks after this cutover require an explicit production maintenance lease, not automatic service stopping.
