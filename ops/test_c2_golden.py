import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import c2_golden as release


class ReleaseConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key_path = self.root / "existing.json"
        self.key_path.write_text(json.dumps({"api_key": "synthetic-test-key-not-a-real-secret"}))
        self.cfg = {"model_name": "qwen3.8-flash-next-iq3_xxs-v0.1.43-golden-c2", "parallel": 2,
                    "cwd": "/qualified-assets", "api_key": "discard-synthetic-benchmark-key", "env": {},
                    "args": ["--spec", "6", "--batch-spec", "6", "--batch-rows", "12",
                             "--kv", "q4_0", "--max-context", "262144", "--vram-reserve-mib", "1400", "--spec-min-p", "0.5",
                             "--batch-pcie-frac", "0", "--pool-workers", "15",
                             "--conversation-cache-mib", "24576", "--conversation-cache-slots", "8",
                             "--conversation-cache-min-free-mib", "8192", "--batch-slot-only", "--batch-mtp",
                             "--batch-expert-merge", "--batch-draft-overlap"]}

    def prepared(self):
        return release.prepare(self.cfg, self.key_path, Path("/qualified/python"), self.root)

    def test_pinned_listener_and_flags(self):
        cfg = self.prepared()
        self.assertEqual((cfg["host"], cfg["port"], cfg["parallel"]), ("0.0.0.0", 8080, 2))
        self.assertEqual(cfg["args"], self.cfg["args"])
        self.assertEqual(cfg["cwd"], self.cfg["cwd"])
        self.assertEqual(cfg["_production"]["binary_sha256"], release.BINARY)
        self.assertFalse(cfg["api_monitor"])
        self.assertFalse(cfg["lazy_load"])
        self.assertEqual(cfg["idle_unload_s"], 0)

    def test_key_never_copied_and_inputs_unchanged(self):
        before = copy.deepcopy(self.cfg)
        old_bytes = self.key_path.read_bytes()
        cfg = self.prepared()
        self.assertNotIn("api_key", cfg)
        self.assertNotIn("synthetic-test-key", json.dumps(cfg))
        self.assertEqual(self.cfg, before)
        self.assertEqual(self.key_path.read_bytes(), old_bytes)

    def test_all_golden_switches(self):
        cfg = self.prepared()
        self.assertEqual(cfg["env"], release.ENV)
        self.assertFalse(set(release.ABSENT).intersection(cfg["env"]))

    def test_reject_diagnostic_even_zero(self):
        self.cfg["env"]["STRATA_VERIFY_PROFILE"] = "0"
        with self.assertRaises(ValueError):
            self.prepared()

    def test_reject_streaming_wrong_dtype_or_reserve(self):
        for flag, value in [("--kv-resident", "32768"), ("--kv", "int8"), ("--vram-reserve-mib", "768")]:
            cfg = copy.deepcopy(self.cfg)
            if flag in cfg["args"]:
                cfg["args"][cfg["args"].index(flag) + 1] = value
            else:
                cfg["args"] += [flag, value]
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                release.validate(cfg)

    def test_reject_capacity_or_other_model(self):
        for change in ({"parallel": 4}, {"model_name": "unqualified-iq3_s"}):
            cfg = {**self.cfg, **change}
            with self.assertRaises(ValueError):
                release.validate(cfg)

    def test_reject_duplicate_and_missing_key(self):
        self.cfg["args"] += ["--spec", "4"]
        with self.assertRaises(ValueError):
            release.validate(self.cfg)
        self.key_path.write_text('{"api_key": ""}')
        with self.assertRaises(ValueError):
            release.key_from(self.key_path)

    def test_run_sanitizes_without_key_argv(self):
        cfg = self.prepared()
        config = self.root / "production.json"
        config.write_text(json.dumps(cfg))
        with patch.object(release, "artifact_check"), patch.object(release, "idle_check"), \
                patch.object(release.os, "chdir"), patch.object(release.os, "execve") as execute, \
                patch.dict(release.os.environ, {"STRATA_VERIFY_PROFILE": "0", "STRATA_API_KEY": "stale",
                                              "STRATA_PRODUCTION_CONFIG_SHA256": release.digest(config)}):
            release.run(config)
        _, argv, env = execute.call_args.args
        self.assertNotIn("--api-key", argv)
        self.assertNotIn("synthetic-test-key", " ".join(argv))
        self.assertNotIn("STRATA_VERIFY_PROFILE", env)
        self.assertEqual(env["STRATA_API_KEY"], "synthetic-test-key-not-a-real-secret")

    def test_unit_process_tree_and_restart(self):
        text = (Path(__file__).parent / "strata-c2-golden.service").read_text()
        for setting in ("Restart=always", "KillMode=control-group", "UMask=0077", "--config",
                        "UnsetEnvironment=STRATA_VERIFY_PROFILE"):
            self.assertIn(setting, text)
        self.assertNotIn("--api-key", text)


if __name__ == "__main__":
    unittest.main()
