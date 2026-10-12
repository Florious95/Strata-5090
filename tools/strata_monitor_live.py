#!/usr/bin/env python3
"""Read-only terminal view of the existing /metrics API; no engine or GPU work."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def key_from(path):
    value = os.environ.get("STRATA_API_KEY")
    if not value:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig")).get("api_key")
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list) or not value or not isinstance(value[0], str) or not value[0].strip():
        raise ValueError("no API key; use --key-config pointing at the existing private config")
    return value[0].strip()


def endpoint(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or
            parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise ValueError("--url must be a server origin without credentials, query or path")
    return url.rstrip("/") + "/metrics"


def fetch(url, key):
    request = urllib.request.Request(endpoint(url), headers={"Authorization": "Bearer " + key})
    # A proxy must not receive a private tailnet/loopback credential.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=3) as response:
        value = json.load(response)
    if not isinstance(value, dict) or not isinstance(value.get("live"), dict):
        raise ValueError("unexpected /metrics schema")
    return value


def safe(value):
    return "".join(c if c.isprintable() else " " for c in str(value))


def number(value, decimals=0):
    return "n/a" if value is None else format(value, f",.{decimals}f")


def render(data, url):
    live, engine = data.get("live", {}), data.get("engine", {})
    hw, recent = data.get("hardware", {}), data.get("requests", [])
    read, total = live.get("prompt_read"), live.get("prompt_total")
    progress = "n/a (no current PP)"
    if read is not None and total:
        filled = min(20, max(0, int(20 * read / total)))
        progress = f"{number(read)}/{number(total)} [" + "■" * filled + "·" * (20 - filled) + "]"
    serving = [f"State: {safe(live.get('state', 'unknown'))}", f"Read {progress}",
               f"Context capacity: {number(engine.get('max_context'))} tokens",
               f"Output: {number(live.get('generated'))} / limit {number(live.get('max_tokens'))}"]
    performance = [f"Decode now: {number(live.get('tok_s'), 1)} tok/s (API rolling window)",
                   f"Request mean: {number(live.get('tok_s_mean'), 1)} tok/s (API)",
                   f"Prefill: {number(live.get('prefill_tok_s_mean'), 1)} tok/s (fresh input)",
                   f"Last native decode: {number(recent[0].get('decode_tok_s'), 1) if recent else 'n/a'} tok/s"]
    lines = [f"STRATA / MONITOR LIVE {safe(url)}  |  {time.strftime('%H:%M:%S')}",
             "READ-ONLY /metrics | fork viewer, NOT the unidentified screenshot CLI", "─" * 112,
             "SERVING".ljust(54) + "PERFORMANCE"]
    for left, right in zip(serving, performance):
        lines.append(left.ljust(54) + right)
    lines += ["─" * 112, "BATCHING / PARALLEL",
              f"Slots: {number(live.get('parallel'))} | running requests: {number(live.get('running'))} | "
              f"queued: {number(live.get('queued'))} | pre-admit: {number(live.get('admitting'))}"]
    for slot in live.get("slots", []):
        lines.append(f"  slot {slot.get('slot')}: {safe(slot.get('state'))} | "
                     f"prompt {number(slot.get('prompt_tokens'))} | output {number(slot.get('generated'))} | "
                     f"API mean {number(slot.get('tok_s'), 1)} tok/s")
    lines += ["─" * 112, "HARDWARE",
              f"GPU busy {number(hw.get('gpu_util'), 1)}% | power {number(hw.get('gpu_power'), 1)} W | "
              f"temperature {number(hw.get('gpu_temp'), 1)} °C",
              f"VRAM {number(hw.get('gpu_mem_used', 0) / 2**30, 2) if hw.get('gpu_mem_used') is not None else 'n/a'} / "
              f"{number(hw.get('gpu_mem_total', 0) / 2**30, 2) if hw.get('gpu_mem_total') is not None else 'n/a'} GiB",
              "─" * 112, "REQUESTS (recent completed; no prompt/answer content)"]
    for i, row in enumerate(recent[:5]):
        lines.append(f"  r{i:02d} prompt {number(row.get('prompt_tokens'))} cached {number(row.get('reused'))} "
                     f"output {number(row.get('output_tokens'))} native {number(row.get('decode_tok_s'), 1)} tok/s "
                     f"finish {safe(row.get('finish'))}")
    lines += ["─" * 112, "ACTIVITY", f"  admission prompt read {progress}",
              "Read = prompt position / request prompt length, NOT output / context capacity.",
              "C2 PP is GLOBAL admission progress; rXX above is recent-list index, not a native request ID.",
              "GPU busy/power are telemetry, NOT DRAM bandwidth utilization or remaining speedup."]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--key-config", type=Path, default=Path.home() / "Projects/Strata/strata-iq3_s.json")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true", help="one frame, suitable for a log/receipt")
    args = parser.parse_args(argv)
    try:
        endpoint(args.url)
        if not math.isfinite(args.interval) or args.interval < 0.2:
            raise ValueError("--interval must be finite and >=0.2 seconds")
        key = key_from(args.key_config)
    except (OSError, ValueError, json.JSONDecodeError):
        parser.error("invalid URL/interval or unreadable private key config; credentials are never printed")
    try:
        while True:
            try:
                frame = render(fetch(args.url, key), args.url)
            except urllib.error.HTTPError as error:
                frame = f"MONITOR UNAVAILABLE: HTTP {error.code}; no stale LIVE data displayed"
            except (OSError, ValueError):
                frame = "MONITOR UNAVAILABLE: network/schema error; no stale LIVE data displayed"
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(frame, flush=True)
            if args.once:
                return 1 if frame.startswith("MONITOR UNAVAILABLE") else 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
