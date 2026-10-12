"""Documentation integrity only; these tests do not qualify runtime/GPU code."""
import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "benchmarks/results/PR-EVIDENCE-LEDGER-20261012.json"
LEDGER = json.loads(LEDGER_PATH.read_text())
IMPORT = json.loads((ROOT / "docs/evidence/ledger-import-verification.json").read_text())
README = (ROOT / "README.md").read_text()


class ForkEvidenceTests(unittest.TestCase):
    def test_archive_bytes(self):
        archive = ROOT / "docs/archive/README.upstream-v0.1.43.md"
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),
                         "415ed3bbd9052b6e7d90a9884e7ff297ee7d1a2248ab6a7ba19ceca28fe02489")
        self.assertEqual(IMPORT["archive_upstream_source_sha"],
                         "7695859aa13afb248bdb4d36f9b2e95707db079d")

    def test_import_hashes(self):
        for path, key in [(LEDGER_PATH, "original_ledger_sha256"),
                          (ROOT / "benchmarks/PR-EVIDENCE-LEDGER-20261012.md",
                           "original_readable_ledger_sha256")]:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), IMPORT[key])
        self.assertEqual(IMPORT["exit_code"], 0)
        self.assertEqual(IMPORT["missing_artifacts"], [])
        self.assertEqual(IMPORT["mismatched_artifacts"], [])

    def test_pooling_and_minima(self):
        self.assertEqual(len(LEDGER["stages"]), 8)
        for stage in LEDGER["stages"]:
            for arm in [stage["ON_or_observation"]] + ([stage["same_build_control"]]
                        if "same_build_control" in stage else []):
                with self.subTest(stage=stage["stage"], source=arm["source_sha"]):
                    cells = arm["cells"]
                    tokens = sum(c["HTTP_tokens"] for c in cells)
                    wall = sum(c["HTTP_wall_s"] for c in cells)
                    self.assertEqual(arm["runs"], len(cells))
                    self.assertEqual(arm["HTTP_tokens"], tokens)
                    self.assertAlmostEqual(arm["HTTP_wall_s"], wall, places=8)
                    self.assertAlmostEqual(arm["pooled_HTTP_tok_s"], tokens / wall, places=8)
                    self.assertAlmostEqual(arm["peak_run_HTTP_tok_s"],
                                           max(c["HTTP_tok_s"] for c in cells), places=8)
                    self.assertEqual(arm["minimum_hit_fraction_across_runs"],
                                     min(c["expert_hit_fraction"] for c in cells))
                    self.assertEqual(arm["minimum_sampled_free_mib"],
                                     min(c["minimum_sampled_free_mib"] for c in cells))

    def test_same_binary_ablations(self):
        controlled = [s for s in LEDGER["stages"] if "same_build_control" in s]
        self.assertEqual(len(controlled), 4)
        for stage in controlled:
            on, off = stage["ON_or_observation"], stage["same_build_control"]
            self.assertEqual(on["source_sha"], off["source_sha"])
            self.assertEqual(on["binary_sha256"], off["binary_sha256"])
            gain = (on["pooled_HTTP_tok_s"] / off["pooled_HTTP_tok_s"] - 1) * 100
            self.assertAlmostEqual(gain, stage["pooled_gain_vs_that_control_pct"], places=8)
            self.assertIn(f"**+{gain:.2f}%**", README)

    def test_readme_figures(self):
        for stage in LEDGER["stages"]:
            on = stage["ON_or_observation"]
            self.assertIn(f'{on["pooled_HTTP_tok_s"]:.2f}', README)
            if stage["stage"] != "Direct Slot":
                self.assertIn(f'{on["peak_run_HTTP_tok_s"]:.2f}', README)
                minimum = f'{on["minimum_hit_fraction_across_runs"] * 100:.3f}% / '
                self.assertIn(minimum + str(int(on["minimum_sampled_free_mib"])), README)

    def test_attribution_boundaries(self):
        self.assertTrue(LEDGER["stage_values_are_NOT_additive_gains"])
        direct = LEDGER["stages"][2]["ON_or_observation"]
        self.assertEqual(direct["runs"], 1)
        self.assertLess(direct["minimum_sampled_free_mib"], 300)
        self.assertFalse(direct["cells"][0]["true_occurrence_counter_valid"])
        for phrase in ["not yet merged", "Chronology, not additive gains",
                       "not a digital signature", "not performance-qualified",
                       "not a paired", "Historical memory OC state is unknown"]:
            self.assertIn(phrase, README)

    def test_local_links(self):
        paths = [ROOT / "README.md", ROOT / "docs/FORK-WORKFLOW.md",
                 ROOT / "docs/FORK-PR-PLAN.md", *sorted((ROOT / "docs/pr").glob("*.md"))]
        for path in paths:
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if "://" in target or target.startswith("#"):
                    continue
                with self.subTest(path=path.name, target=target):
                    self.assertTrue((path.parent / target.split("#", 1)[0]).exists())

    def test_official_observation_limits(self):
        obs = json.loads((ROOT / "docs/evidence/official43-observation.json").read_text())
        self.assertFalse(obs["performance_qualified"])
        self.assertFalse(obs["production_deployed"])
        for arm in obs["arms"]:
            self.assertEqual(arm["benchmark_exit"], 1)
            self.assertFalse(arm["performance_qualified"])
            self.assertTrue(arm["failed_observer_gates"])
            self.assertAlmostEqual(arm["HTTP_tok_s"],
                                   arm["completion_tokens"] / arm["HTTP_wall_s"], places=8)
            self.assertIn(f'{arm["HTTP_tok_s"]:.2f}', README)

    def test_original_hash_index_coverage(self):
        index = IMPORT["checked_original_artifacts_sha256"]
        self.assertEqual(len(index), IMPORT["unique_original_artifacts_checked"])
        self.assertGreaterEqual(len(index), 154)

        def walk(node):
            if isinstance(node, dict):
                for name_key, hash_key in [("summary_receipt", "summary_sha256"),
                                           ("receipt", "receipt_sha256")]:
                    if name_key in node and hash_key in node:
                        self.assertEqual(index[node[name_key]], node[hash_key])
                for key, value in node.items():
                    if key in ("artifacts_sha256", "negative_and_diagnostic_receipts_retained",
                               "new_C1_C2_NVML_receipts"):
                        for name, digest in value.items():
                            self.assertEqual(index[name], digest)
                            self.assertRegex(digest, r"^[a-f0-9]{64}$")
                    else:
                        walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(LEDGER)


if __name__ == "__main__":
    unittest.main()
