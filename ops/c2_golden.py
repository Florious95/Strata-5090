#!/usr/bin/env python3
"""Pinned C2 release config, preflight and authenticated smoke. Never print keys."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

SOURCE = "77cd2b2de043721bda8deff2fd691e88d7499a2a"
BINARY = "640c5a6600e687c23c870a99d16b3c59bd769c86aefd0b331460371fa1c950d8"
SERVER = "06ca08c2497f4200a46d602b88e8b87df91e79b8fc2af0caf855b4da7358ddb1"
CANDIDATE = Path("/home/florious/Projects/Strata-c2-v0.1.43-port")
ENV = {"STRATA_BATCH_DENSE_PARALLEL": "1", "STRATA_BATCH_NARROW_PARALLEL": "1",
       "STRATA_BATCH_SHARED_AFTER_RING": "1", "STRATA_BATCH_HEAD_WIDE": "1",
       "STRATA_BATCH_QSA_TILE": "1", "STRATA_TOKEN_GAP_RAW": "1",
       "STRATA_PLE_KEYED_PREFETCH": "0", "STRATA_NAN_GUARD": "1",
       "STRATA_ROUTE_TAIL_SKIP": "0"}
ABSENT = ("STRATA_VERIFY_DEVICE_PLAN", "STRATA_VERIFY_PROFILE", "STRATA_BATCH_DENSE_PROFILE",
          "STRATA_BATCH_HOST_PROFILE", "STRATA_BATCH_RING_PROFILE", "STRATA_NAN_INJECT",
          "STRATA_DBG_NAN", "STRATA_ROUTE_PRIOR", "STRATA_ROUTE_RESIDENT",
          "STRATA_ROUTE_RESIDENT_MTP", "STRATA_API_KEY")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), "config must be an object")
    return value


def key_from(path):
    value = load(path).get("api_key")
    keys = value.split(",") if isinstance(value, str) else value
    require(isinstance(keys, list) and bool(keys), "existing production API key is required")
    require(all(isinstance(k, str) and len(k.strip()) >= 16 for k in keys),
            "existing production keys must be nonempty strong secrets; do not print them")
    return ",".join(k.strip() for k in keys)


def option(args, flag, default=None):
    places = [i for i, a in enumerate(args) if a == flag]
    require(len(places) <= 1, "duplicate critical option: " + flag)
    if not places:
        return default
    i = places[0]
    require(i + 1 < len(args) and not args[i + 1].startswith("--"), "missing value: " + flag)
    return args[i + 1]


def validate(cfg):
    args = cfg.get("args")
    require(isinstance(args, list) and all(isinstance(x, str) for x in args), "args must be strings")
    for flag, value in {"--spec": "6", "--batch-spec": "6", "--batch-rows": "12",
                        "--kv": "q4_0", "--vram-reserve-mib": "1400",
                        "--batch-pcie-frac": "0", "--pool-workers": "15",
                        "--conversation-cache-mib": "24576", "--conversation-cache-slots": "8",
                        "--conversation-cache-min-free-mib": "8192"}.items():
        require(option(args, flag) == value, "frozen option mismatch: " + flag)
    batch = option(args, "--batch", option(args, "--slots", str(cfg.get("parallel", 1))))
    require(batch == "2", "exactly two active batch slots are required")
    require(option(args, "--kv-resident", "0") == "0", "C2 streaming is not qualified")
    for flag in ("--batch-slot-only", "--batch-mtp", "--batch-expert-merge", "--batch-draft-overlap"):
        require(flag in args, "missing qualified option: " + flag)
    require(float(option(args, "--spec-min-p", "0")) == 0.5, "draft confidence threshold must remain 0.5")
    require("--batch-spec-fixed" not in args and "--no-host-worker" not in args,
            "unqualified fixed layout or host-worker change")
    require("iq3_xxs" in str(cfg.get("model_name", "")).lower(), "release model must remain IQ3_XXS")
    require(not cfg.get("vision_on_demand") and "--vision-on-demand" not in args,
            "standalone vision-on-demand is not implemented")
    require("--mmproj" not in args, "mmproj belongs in the vision section, not text-engine args")
    vision = cfg.get("vision")
    if vision:
        require(isinstance(vision, dict), "vision must be an object")
        require(vision.get("gpu", False) is False, "this release qualifies CPU vision only")
        require(all(isinstance(vision.get(k), str) and vision[k] for k in ("exe", "mmproj", "model")),
                "vision executable, projector and vocabulary model are required")
        for field in ("threads", "max_tokens", "min_tokens"):
            if field in vision:
                require(type(vision[field]) is int and vision[field] > 0, "positive integer vision setting required")
        if "min_tokens" in vision and "max_tokens" in vision:
            require(vision["min_tokens"] <= vision["max_tokens"], "vision minimum exceeds token cap")
    env = cfg.get("env") or {}
    require(isinstance(env, dict), "env must be an object")
    require(not any(k in env for k in ABSENT), "diagnostic/approximation/key environment is forbidden")
    if "--context" in args:
        require(option(args, "--context") == "262144", "context must remain 262144")
    elif "--max-context" in args:
        require(option(args, "--max-context") == "262144", "context must remain 262144")
    else:
        raise ValueError("an explicit 262144 context is required")


def prepare(cfg, key_path, python, state):
    validate(cfg)
    key_from(key_path)  # Validate in memory; never copy the secret into the new config.
    cfg = dict(cfg)
    cfg.pop("api_key", None)
    cfg.pop("before_load", None)
    cfg.pop("mcp_servers", None)
    cfg.update(exe=str(CANDIDATE / "build/strata"), host="0.0.0.0", port=8080, parallel=2,
               lazy_load=False, idle_unload_s=0, open_browser=False, api_monitor=False,
               cors_origins=[], trusted_origins=[], log=str(state / "engine.log"))
    cfg["env"] = {**(cfg.get("env") or {}), **ENV}
    cfg["_production"] = {"source_sha": SOURCE, "binary_sha256": BINARY,
                          "server_sha256": SERVER, "key_config": str(key_path), "python": str(python)}
    return cfg


def artifact_check(cfg):
    validate(cfg)
    meta = cfg.get("_production") or {}
    require(meta.get("source_sha") == SOURCE and meta.get("binary_sha256") == BINARY and
            meta.get("server_sha256") == SERVER, "release metadata mismatch")
    require(cfg.get("exe") == str(CANDIDATE / "build/strata"), "wrong engine path")
    require(cfg.get("host") == "0.0.0.0" and cfg.get("port") == 8080 and cfg.get("parallel") == 2,
            "production listener/capacity mismatch")
    require(all((cfg.get("env") or {}).get(k) == v for k, v in ENV.items()), "golden environment mismatch")
    require("api_key" not in cfg, "new config must not duplicate the existing secret")
    require(digest(cfg["exe"]) == BINARY, "binary SHA mismatch")
    require(digest(CANDIDATE / "serve/server.py") == SERVER, "server SHA mismatch")
    head = subprocess.check_output(["git", "-C", str(CANDIDATE), "rev-parse", "HEAD"], text=True).strip()
    require(head == SOURCE, "candidate source SHA mismatch")
    for staged in ([], ["--cached"]):
        subprocess.run(["git", "-C", str(CANDIDATE), "diff", "--quiet", *staged, "HEAD", "--",
                        "src", "include", "serve", "tools"], check=True)
    require(Path(meta["python"]).is_file() and os.access(meta["python"], os.X_OK), "Python venv missing")
    require(Path(cfg.get("cwd", "")).is_absolute() and Path(cfg["cwd"]).is_dir(), "asset cwd missing")
    tokenizer = Path(cfg.get("tokenizer", ""))
    require(tokenizer.is_absolute() and all((tokenizer / f).is_file()
            for f in ("vocab.json", "merges.txt", "token_type.json")), "absolute tokenizer assets missing")
    for flag in ("--pack", "--mtp", "--native-preset", "--native-dense-gguf", "--ple-gguf", "--expert-profile"):
        for i, a in enumerate(cfg["args"]):
            if a == flag:
                require(i + 1 < len(cfg["args"]), "asset option value missing")
                p = Path(cfg["args"][i + 1])
                if not p.is_absolute():
                    p = Path(cfg["cwd"]) / p
                require(p.exists(), "qualified asset missing for " + flag)
    for field, value in (cfg.get("vision") or {}).items():
        if field not in ("exe", "mmproj", "model"):
            continue
        p = Path(value)
        if not p.is_absolute():
            p = Path(cfg["cwd"]) / p
        require(p.is_file(), "CPU vision asset missing: " + field)
        if field == "exe":
            require(os.access(p, os.X_OK), "vision executable is not runnable")
        else:
            with p.open("rb") as f:
                require(f.read(4) == b"GGUF", "vision asset is not GGUF: " + field)
    require(os.path.ismount("/mnt/strata-models"), "model filesystem is not mounted")
    key_from(meta["key_config"])


def idle_check():
    pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                   text=True).strip()
    require(not pids, "GPU compute process exists; stop only the owned old/benchmark unit, never kill unknown PIDs")
    used = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   text=True).strip()
    require(int(used) <= 64, "GPU is not clean; investigate non-compute/graphics occupancy")
    with socket.socket() as s:
        s.bind(("0.0.0.0", 8080))


def run(config):
    require(digest(config) == os.environ.get("STRATA_PRODUCTION_CONFIG_SHA256"), "production config SHA mismatch")
    cfg = load(config)
    artifact_check(cfg)
    idle_check()
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    # No stale community/benchmark knobs from the user manager; child_env adds the frozen config's env.
    env.update(ENV)
    env["STRATA_API_KEY"] = key_from(cfg["_production"]["key_config"])
    os.chdir(CANDIDATE)
    python = cfg["_production"]["python"]
    os.execve(python, [python, "-m", "serve.server", "--engine", "strata", "--config", str(config),
                       "--host", "0.0.0.0", "--port", "8080"], env)


def smoke(config, base, timeout):
    cfg = load(config)
    key = key_from(cfg["_production"]["key_config"]).split(",")[0]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, body=None, auth=False):
        headers = {"Authorization": "Bearer " + key} if auth else {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(base.rstrip("/") + path,
                                     data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try:
            with opener.open(req, timeout=timeout) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            e.read()
            return e.code, None

    deadline = time.monotonic() + timeout
    while True:
        try:
            code, health = request("/health")
            if code == 200 and health.get("loaded") is True:
                require(health.get("api_key") is True, "server authentication is not enabled")
                break
        except (OSError, json.JSONDecodeError):
            pass
        require(time.monotonic() < deadline, "model did not become healthy within deadline")
        time.sleep(1)
    require(request("/v1/models")[0] == 401, "unauthenticated models must return 401")
    require(request("/v1/models", auth=True)[0] == 200, "authenticated models must return 200")

    def chat(i):
        status, data = request("/v1/chat/completions", {
            "model": cfg["model_name"], "messages": [{"role": "user", "content":
                f"Write a short numbered checklist for code review and tests. Checklist {i}."}],
            "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, True)
        require(status == 200 and data and data.get("choices"), "production chat smoke failed")
        require(any((c.get("message") or {}).get("content") for c in data["choices"]), "empty chat output")
        return (data.get("usage") or {}).get("completion_tokens", 0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        tokens = list(pool.map(chat, (1, 2)))
    print(json.dumps({"loaded": True, "unauth_models": 401, "auth_models": 200,
                      "concurrent_chat_requests": 2, "completion_tokens": tokens,
                      "scope": "API smoke, not full258K acceptance or proof of dual GPU occupancy"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--golden-config", type=Path, required=True)
    p.add_argument("--golden-config-sha256", required=True)
    p.add_argument("--key-config", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("check", "run", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name == "check":
            p.add_argument("--idle", action="store_true")
        if name == "smoke":
            p.add_argument("--base", default="http://127.0.0.1:8080")
            p.add_argument("--timeout", type=int, default=240)
    a = ap.parse_args()
    try:
        if a.command == "prepare":
            require(a.golden_config.is_absolute() and a.key_config.is_absolute() and a.python.is_absolute()
                    and a.output.is_absolute(), "all deployment paths must be absolute")
            require(digest(a.golden_config) == a.golden_config_sha256, "frozen config SHA mismatch")
            state = Path.home() / ".local/state/strata-c2-golden"
            state.mkdir(parents=True, exist_ok=True, mode=0o700)
            cfg = prepare(load(a.golden_config), a.key_config, a.python, state)
            cfg["_production"]["golden_config_sha256"] = a.golden_config_sha256
            artifact_check(cfg)
            a.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            integrity = a.output.parent / "release.env"
            require(not a.output.exists() and not integrity.exists(), "production config already exists; back up before replacement")
            with a.output.open("x", encoding="utf-8") as f:
                os.chmod(a.output, 0o600)
                json.dump(cfg, f, indent=2)
                f.write("\n")
            with integrity.open("x", encoding="utf-8") as f:
                os.chmod(integrity, 0o600)
                f.write("STRATA_PRODUCTION_CONFIG_SHA256=" + digest(a.output) + "\n")
            print("prepared config_sha256=" + digest(a.output) + "; secret remains in existing target key config")
        elif a.command == "check":
            artifact_check(load(a.config))
            if a.idle:
                idle_check()
            print("preflight=PASS")
        elif a.command == "run":
            run(a.config.resolve())
        else:
            smoke(a.config, a.base, a.timeout)
        return 0
    except Exception as e:
        # Do not render exception payloads from HTTP/private config or credential-bearing fields.
        print("FAIL: deployment validation/operation rejected; " + type(e).__name__, file=sys.stderr)
        if isinstance(e, ValueError):
            print(str(e), file=sys.stderr)  # only fixed messages written above, never config/key values
        return 2


if __name__ == "__main__":
    sys.exit(main())
