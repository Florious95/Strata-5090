"""CPU-only contracts: display raw API facts and never manufacture chart values."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import xml.etree.ElementTree as ET

from tools import render_throughput_comparison as chart
from tools import strata_monitor_live as monitor
from tools import render_tui_report as tui


class MonitorTests(unittest.TestCase):
    def test_read_capacity_output_and_timer_are_separate(self):
        frame = monitor.render({"engine": {"max_context": 262144},
            "live": {"state": "reading", "prompt_read": 250, "prompt_total": 400,
                     "generated": 23, "max_tokens": 100, "parallel": 2}}, "http://127.0.0.1:8080")
        self.assertIn("Read 250/400", frame)
        self.assertIn("Context capacity: 262,144", frame)
        self.assertIn("Output: 23 / limit 100", frame)
        self.assertIn("n/a tok/s", frame)
        self.assertNotIn("250/262,144", frame)
        self.assertIn("GLOBAL admission", frame)

    def test_long_progress_uses_exact_numbers_no_k(self):
        frame = monitor.render({"live": {"prompt_read": 87000, "prompt_total": 258000}}, "http://localhost")
        self.assertIn("87,000/258,000", frame)
        self.assertNotIn("87K", frame)
        self.assertNotIn("258K", frame)

    def test_control_characters_are_not_rendered(self):
        frame = monitor.render({"live": {"state": "\x1b[2J\n"}}, "http://localhost")
        self.assertNotIn("\x1b", frame)
        self.assertEqual(monitor.safe("a\nb"), "a b")

    def test_unknowns_do_not_become_zero_or_context_progress(self):
        frame = monitor.render({"live": {"state": "idle"}, "engine": {"max_context": 262144}}, "http://localhost")
        self.assertIn("no current PP", frame)
        self.assertIn("GPU busy n/a%", frame)
        self.assertIn("VRAM n/a / n/a", frame)

    def test_auth_and_get_only_no_prompt_body_or_key_display(self):
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append((self.path, self.headers.get("Authorization")))
                body = json.dumps({"live": {"state": "idle"}}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            data = monitor.fetch(f"http://127.0.0.1:{server.server_port}", "fixture-secret")
            self.assertEqual(calls, [("/metrics", "Bearer fixture-secret")])
            self.assertNotIn("fixture-secret", monitor.render(data, "http://localhost"))
        finally:
            server.shutdown(); server.server_close(); worker.join()

    def test_invalid_origins_and_error_frame(self):
        for url in ("file:///x", "http://user:secret@localhost", "http://localhost/v1", "http://localhost?k=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                monitor.endpoint(url)
        output = io.StringIO()
        with mock.patch.object(monitor, "key_from", return_value="fixture-secret"), \
                mock.patch.object(monitor, "fetch", side_effect=OSError("fixture-secret")), redirect_stdout(output):
            self.assertEqual(monitor.main(["--once"]), 1)
        self.assertIn("MONITOR UNAVAILABLE", output.getvalue())
        self.assertNotIn("fixture-secret", output.getvalue())
        self.assertNotIn("LIVE", output.getvalue().replace("stale LIVE", "stale"))

    def test_snapshot_is_allowlisted_and_keeps_capture_provenance(self):
        data = {"live": {"prompt_read": 250, "prompt_total": 400, "api_key": "do-not-save",
                         "slots": [{"slot": 1, "prompt": "private-content", "state": "reading"}]},
                "requests": [{"output": "private-output", "output_tokens": 50}],
                "api_key": "do-not-save", "hardware": {"gpu_pcie_gen": 5, "gpu_pcie_width": 16},
                "hardware_static": {"gpu_count": 1, "gpu_name": "NVIDIA RTX 5090"}}
        saved = monitor.snapshot(data, "http://localhost:8080")
        self.assertNotIn("do-not-save", json.dumps(saved))
        self.assertNotIn("private-", json.dumps(saved))
        self.assertEqual(saved["live"]["prompt_total"], 400)
        self.assertTrue(saved["captured_at"])
        self.assertEqual(saved["source_endpoint"], "http://localhost:8080/metrics")
        frame = monitor.render(saved, "http://localhost:8080")
        self.assertIn("32.0 GT/s ×16 (not GDDR MT/s)", frame)
        self.assertIn(saved["captured_at"], frame)

    def test_snapshot_requires_once(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            monitor.main(["--snapshot", "/tmp/no-write.json"])
        self.assertEqual(caught.exception.code, 2)

    def test_private_key_config_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.json"
            path.write_text(json.dumps({"api_key": ["first-secret", "second-secret"]}))
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(monitor.key_from(path), "first-secret")
            with mock.patch.dict(os.environ, {"STRATA_API_KEY": "env-secret,other"}):
                self.assertEqual(monitor.key_from(path), "env-secret")


class ChartTests(unittest.TestCase):
    def args(self, **changes):
        values = dict(short_tok_s=None, short_source=None, short_basis=None, short_prompt_tokens=250,
                      community_tok_s=None, community_source=None, community_basis=None)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_pending_is_none_not_zero_and_basis_preserved(self):
        items = chart.rows(self.args())
        self.assertEqual([r["value"] for r in items], [209.9903, 173.4, None, None])
        self.assertIn("HTTP pooled", items[0]["basis"])
        self.assertIn("native decode", items[1]["basis"])

    def test_finite_positive_only(self):
        for value in ("nan", "inf", "-inf", "0", "-1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                chart.positive(value)
        self.assertEqual(chart.positive("300.25"), 300.25)

    def test_values_require_source_and_timer(self):
        for values in (dict(short_tok_s=300), dict(community_tok_s=1000),
                       dict(short_tok_s=300, short_source="fixture.json"), dict(short_prompt_tokens=75000)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                chart.rows(self.args(**values))

    def test_svg_pending_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison"
            chart.draw(chart.rows(self.args()), output, svg_only=True)
            ET.parse(output.with_suffix(".svg"))
            svg = output.with_suffix(".svg").read_text()
            self.assertIn("待实测", svg); self.assertIn("待来源", svg)
            self.assertIn("209.99", svg); self.assertIn("173.40", svg)
            data = json.loads(output.with_suffix(".json").read_text())
            self.assertIsNone(data["rows"][2]["value"])
            self.assertFalse(data["directly_comparable"])
            self.assertEqual(data["axis"], "linear-zero-origin")
            self.assertFalse(output.with_suffix(".png").exists())

    def test_supplied_values_are_labeled_and_xml_escaped(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(short_tok_s=300.25, short_source="fixture-only.json", short_basis="fixture <peak>& native C1",
                community_tok_s=1000.5, community_source="https://example.invalid/fixture", community_basis="fixture claim,4GPU")
            output = Path(directory) / "fixture-not-measurement"
            chart.draw(chart.rows(args), output, svg_only=True)
            ET.parse(output.with_suffix(".svg"))
            svg = output.with_suffix(".svg").read_text()
            self.assertIn("300.25", svg); self.assertIn("1000.50", svg)
            self.assertIn("&lt;peak&gt;&amp;", svg)
            self.assertIn("fixture-only.json", output.with_suffix(".json").read_text())

    def test_cli_rejects_bad_measurement_without_writing_chart(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            output = Path(directory) / "bad"
            with self.assertRaises(SystemExit) as caught:
                chart.main(["--short-tok-s", "300", "--svg-only", "--output", str(output)])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(list(Path(directory).iterdir()), [])


class TuiTests(unittest.TestCase):
    def test_render_measured_snapshot_and_separate_peak(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture-snapshot.json"
            output = Path(directory) / "fixture-only.png"
            data = monitor.snapshot({"live": {"prompt_read": 250, "prompt_total": 400},
                                     "hardware_static": {"gpu_name": "fixture GPU", "gpu_count": 1}}, "http://localhost:8080")
            source.write_text(json.dumps(data))
            width, height = tui.draw(source, output, peak=300.25, source="fixture-only", basis="fixture peak")
            with Image.open(output) as image:
                self.assertEqual(image.size, (width, height))
                self.assertEqual(width, 2400)
                self.assertGreater(height, 1200)
            meta = json.loads(output.with_suffix(".json").read_text())
            self.assertTrue(meta["rendered_not_terminal_capture"])
            self.assertEqual(meta["peak_tok_s"], 300.25)
            self.assertEqual(len(meta["snapshot_sha256"]), 64)
            self.assertEqual(json.loads(source.read_text())["live"]["prompt_total"], 400)

    def test_tui_rejects_unbound_or_unsourced_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.json"
            output = Path(directory) / "not-created.png"
            source.write_text('{"live":{}}')
            with self.assertRaises(ValueError):
                tui.draw(source, output)
            source.write_text(json.dumps(monitor.snapshot({"live": {}}, "http://localhost:8080")))
            with self.assertRaises(ValueError):
                tui.draw(source, output, peak=300)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
