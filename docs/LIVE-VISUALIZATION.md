# Real terminal panel and throughput comparison

## Scope / task contract

- Goal: display real production facts immediately and render the requested high-resolution comparison without inventing missing benchmark values.
- Base: `1668d83af65bb6bf7c8141b2f0eacec758d0627d` (README/evidence PR#1); this optional tooling PR does not change engine, server, release pins, services, credentials, sampling or GPU workload.
- The screenshot's external `strata monitor` entry is **not present** in official v0.1.43 or the frozen C2 tree/PATH. This viewer is an explicitly identified fork replacement, not a claim that its renderer was found.
- Source facts come only from authenticated GET `/metrics`. No generation, prefill, loading, service restart or GPU work is performed. Independent production acceptance remains bench-owned.

## Immediately usable panel

Use a **separate observer clone**, not the delivery clone whose checkout supplies the running systemd launcher:

```bash
git clone --branch docs/live-monitor-throughput-comparison-20261012 https://github.com/Florious95/Strata-5090.git ~/Projects/Strata-5090-observe
```

Then the complete target-local invocation is:

```bash
python3 ~/Projects/Strata-5090-observe/tools/strata_monitor_live.py --url http://127.0.0.1:8080 --key-config ~/Projects/Strata/strata-iq3_s.json
```

Add `--once` for one evidence frame. Add `--once --snapshot /absolute/public-frame.json` to archive only allowlisted numeric/display fields with capture time; no arbitrary API extensions or request content. Standard library only. The key is read inside the target process; never put it in argv, output or Git. The monitor bypasses inherited HTTP proxies so a tailnet/loopback credential is not sent to a proxy. It shows no prompts/answers. Ctrl-C exits the **viewer only**.

Panels: SERVING, PERFORMANCE, BATCHING / PARALLEL, HARDWARE, REQUESTS, ACTIVITY. Read is displayed as exact integers, e.g. `87,000/258,000`; configured capacity262,144 is a separate row. C2 PP is global admission progress, **not** a per-slot request ID. No active prompt means `n/a`, not a fabricated zero progress bar. The recent list's `r00` is an index, not a native request identifier.

API rolling-window/whole-request speed and the last native decode speed are labeled separately. GPU busy/power do not measure bandwidth saturation or remaining headroom. An unavailable API explicitly replaces the frame; stale data is not labeled LIVE. It is a1Hz snapshot, not a nanosecond trace; very short prefill can complete between samples.

For the original official web panel, open `http://100.99.91.50:8080` and select Monitor with the existing credential. This does not supply the unidentified screenshot CLI. Never substitute API screenshot speed for a pooled benchmark receipt.

## PNG + SVG comparison

`tools/render_throughput_comparison.py` uses the existing Pillow dependency for PNG; SVG-only requires no third-party package. It emits2400×1620 PNG (200dpi), SVG and a JSON containing exact values, sources and timing bases. A Chinese font is needed for PNG; macOS Heiti, Linux Noto Sans CJK and Windows YaHei are auto-selected, or use `--font`.

Pending preview already generated: [PNG](media/throughput-comparison-pending.png), [SVG](media/throughput-comparison-pending.svg), [metadata](media/throughput-comparison-pending.json).

```bash
python3 tools/render_throughput_comparison.py --output docs/media/throughput-comparison-pending
```

After bench supplies real measurements (shell variables below represent receipts, **not example numbers**):

```bash
python3 tools/render_throughput_comparison.py \
  --short-tok-s "$SHORT_PEAK_TOK_S" --short-prompt-tokens "$ACTUAL_PROMPT_TOKENS" \
  --short-source "$SHORT_RECEIPT" --short-basis "$SHORT_TIMER_ACTIVE_SLOTS_OUTPUT_STATISTIC" \
  --community-tok-s "$FOUR_GPU_CLAIM" --community-source "$PUBLIC_CLAIM_URL" \
  --community-basis "$FOUR_GPU_HARDWARE_CONTEXT_TIMER_QUALITY" \
  --output docs/media/throughput-comparison-measured
```

Unknown community data can be omitted while the measured short result is rendered. Unknown values are **None/pending**, not zero bars. Non-finite/non-positive throughput, a value without source/timer, or a toy prompt outside250..400tokens fails before output. Test fixture numbers live only in temporary test directories, never the published plot.

The210 user-rounded point is historical C2 HTTP pooled209.9903 (24,576/117.033979s), not two lanes each210 or a new production measurement. The173 user-rounded point uses the latest75,680-token production request, native173.4 (1005output/5.7953s), one active slot. The broader55-round native pool172.977 includes contexts9,989→75,680 and is annotated separately; it is **not** a fixed75K pool. Approximate service-active rates164.75/160.59 and tool/idle interval35.22 are different timers and are not substituted.

The short input is an actual prompt length, not configured capacity; a long output grows KV beyond its initial250tokens. A selected peak is not median/pooled speed. Community4-card values remain external claims with their own context, GPU models, timer and quality settings. Every known bar shares a linear zero-origin axis; no broken/log axis or speedup claim hides the incomparable conditions. A high short-context rate is not intrinsically fake—misrepresenting it as long-context business performance would be misleading.

## Receipt-backed TUI PNG war report

After bench captures a real frame (separate observer clone; the key remains target-local):

```bash
python3 ~/Projects/Strata-5090-observe/tools/strata_monitor_live.py --once \
  --key-config ~/Projects/Strata/strata-iq3_s.json --snapshot /absolute/public-frame.json
python3 tools/render_tui_report.py --snapshot /absolute/public-frame.json \
  --peak-tok-s "$ACTUAL_PEAK" --peak-source "$RAW_RECEIPT" --peak-basis "$TIMER_AND_STATISTIC" \
  --output docs/media/5090-tui-measured.png
```

The dark2400px-wide report uses the SERVING/PERFORMANCE/BATCHING/HARDWARE/REQUESTS/ACTIVITY layout, exact observed Read, GPU name/count/VRAM and measured PCIe generation/width. PCIe5 means32GT/s per lane, **not** the user-reported32Gb/s GDDR data rate. Missing hardware is n/a. Snapshot capture time/input SHA and rendered-not-terminal-capture labeling remain visible. A selected benchmark peak is a **separate** annotation, never substituted for snapshot PERFORMANCE. No hard-codedRead250/400 if actual PP differs or is unavailable. The unidentified original CLI is not claimed pixel-exact.

The first reported short result is [PNG](media/throughput-comparison-c2-short-reported.png)/[SVG](media/throughput-comparison-c2-short-reported.svg):315-token C2 production,292.45 HTTP aggregate and313.0 two-second rolling peak, **forwarded by Leader from bench**, raw receipt still pending. The plotted toy point is313.0 and its basis also records292.45. Community4600 is user-relayed4-GPU claim with context/timer/hardware/quality unverified; the plot does not assert that4600 was measured on250tokens. Future higher results must retain separate receipt/timer metadata.

## Safe overnight short-context matrix (source77cd/c15; no code changes)

Use a target-private experimental config/loopback8081, not the strict C2 production launcher. Backup the **actually deployed** source/binary/config/ops/owned unit before stopping it under the user-authorized overnight maintenance. Do not change IQ3_XXS assets or export credentials. Text-only benchmark removesvision; C>2 also removes`--batch-slot-only`. Keep full Q4 and explicit`--batch-groups 1` on this single GPU.

| Arm | Key arguments | Physical limit |
|---|---|---|
| C2 golden | `--batch 2 --spec 6 --batch-spec 6 --batch-rows 12 --batch-slot-only` | Full DirectSlot/adaptive/Narrow stack |
| C4 packed12 | `--batch 4 --spec 4 --batch-spec 4 --batch-rows 12 --batch-expert-merge --batch-draft-overlap` | At4active, at most3rows/slot; widehead9..12 + QSA/Dense when actual guards hold; no DirectSlot/adaptive/Narrow-only-C2 |
| C4 packed16 | Same, rows16 | At4active4rows/slot; Dense8+8; Q5 wide single-read9..12 **does not** cover16, so head reads8+8. NaN flag buffer is dynamicallymax_t sized. Not presumed faster. |
| C6/C8 plain | `--batch 6` or`8`, `--spec 4 --batch-spec 1 --batch-rows 8`, no`--batch-mtp` | Native one-token/slot; batch_fast is off atC>4; do not call this wide speculative stack |
| C6/C8 legacyMTP control | Add`--batch-mtp` to preceding arm only | Up to8rows/window: at most4slots×2rows per wave, not all8slots×2 simultaneously |

Start`--max-context 2048`, actualprompt250..400 andmax_tokens1024. Atcontext1024/prompt315, max_new≤701 (`n+max_new+8≤ctx`); use512, not1024. A315-token initial input grows KV during decode; it is not constant315-context compute.

Keep the five golden ENV toggles requested but report actual nativeactive/rows/dense/head/qsa dispatch rather than assuming all are active. Profiles must be **absent**, NaN guard1, keyed prefetch0. `TAIL_SKIP=7` is an authorized **quality-tradeoff** miss-tail control (zero-basedrank≥7, at mostlast3missing experts), not an excuse to bypassnonfinite/error detection. It may do little if short-context expertcache hit is near100%.

First batch-pcie-frac0/pool15; then independent0.75/pool7 sweep. `batch-pcie-frac` overrides globalpcie in fast mode; plain C6/C8 uses globalpcie, so record which one is effective. Sweep spec/probability/suffix only with effective-path receipts; repetition or lookup favorable inputs are disclosed, not marketed as general coding. No four-digit throughput promise. Rawroute/skip/token counts, actualactive, HTTP pool versusrolling peak, contextgrowth, timers and minVRAM are retained.

Morning restoration is bench-owned and restores the saved **complete actual production release** after experimentPID/cgroup/GPU clears. Do not blindly start oldofficialC1, force unacceptedvision, or mix77source withc15binary. Originalruntime has hardC2DirectSlot andC≤4packed boundaries; removing them needs another isolated tested implementation, not just parameters.

## Verification / handoff

```bash
python3 -m unittest tools.test_live_visualization -v
python3 -m unittest discover -s tools -p test_fork_evidence.py -v
python3 tools/render_throughput_comparison.py --svg-only --output /tmp/comparison-pending
```

The CPU contracts cover authenticated GET only, no key leakage, exact progress versus capacity/output, unknowns/stale-data failure, mandatory source/timing, numeric rejection, XML escaping and pending metadata. Local PNG is also generated and inspected. GitHub CI is scoped to these contracts and the existing evidence pack; it is not a live production/GPU test.

Do not install a new global `strata` alias or switch the live delivery checkout to this branch. Fetch the observer through Git; bench runs the panel while owning production. Await short-context/4-card receipts before rendering final measured artifacts. No self-merge.
