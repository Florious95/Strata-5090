"""CPU protocol/observability qualification. These are not GPU or model-quality evidence."""
import concurrent.futures
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, _TokenGaps, request_stats, request_timings
from serve.test_parallel import FAKE_BATCH

FAKE_SLOT_ONLY = FAKE_BATCH.replace("INFO engine=0.1.39", "INFO engine=0.1.39 slot_only=1").replace(
    '    if line.startswith(("GEN ", "BGEN ")):\n',
    '    if line.startswith("GEN "):\n'
    '        print("ERR slot-only received solo GEN", flush=True)\n'
    '        continue\n'
    '    if line.startswith(("GEN ", "BGEN ")):\n',
).replace('            print(f"BDONE {b}', '            time.sleep(0.08)\n            print(f"BDONE {b}')


class TokenGaps(unittest.TestCase):
    def test_empty_and_single_token_have_no_gap(self):
        g = _TokenGaps()
        self.assertEqual(g.receipt()["token_gap_count"], 0)
        self.assertIsNone(g.receipt()["token_gap_p95_ms"])
        g.observe(0)
        self.assertEqual(g.receipt()["token_gap_tokens"], 1)
        self.assertIsNone(g.receipt()["token_gap_p95_ms"])

    def test_exact_arrivals_and_timings_export(self):
        g = _TokenGaps()
        for t in (0, 1_000_000, 3_000_000, 103_000_000):
            g.observe(t)
        receipt = g.receipt()
        self.assertEqual(receipt["token_gap_count"], 3)
        self.assertEqual(receipt["token_gap_p95_ms"], 100.0)
        last = {"prompt_ms": 1, "decode_ms": 2, "generated": 4, **receipt}
        timings = request_timings(20, 4, last)
        self.assertEqual(timings["token_gap_source"], "engine_stdout_token_id")
        self.assertEqual(timings["token_gap_tokens"], 4)


    def test_raw_receipts_merge_across_continuations(self):
        first, second = _TokenGaps(), _TokenGaps()
        for t in (0, 1_000_000): first.observe(t)
        for t in (9_000_000, 11_000_000): second.observe(t)
        with mock.patch.dict(os.environ, {"STRATA_TOKEN_GAP_RAW": "1"}):
            merged = request_stats([first.receipt(), second.receipt()])
        self.assertEqual(merged["token_gaps_ms"], [1.0, 8.0, 2.0])
        self.assertEqual(merged["token_gap_count"], 3)
        self.assertEqual(merged["token_gap_tokens"], 4)
        self.assertEqual(merged["token_gap_p95_ms"], 8.0)
        self.assertEqual(merged["token_gap_first_ns"], 0)
        self.assertEqual(merged["token_gap_last_ns"], 11_000_000)

    def test_default_receipt_is_bounded_and_multiple_p95s_not_fabricated(self):
        g = _TokenGaps()
        g.observe(0)
        g.observe(2_000_000)
        with mock.patch.dict(os.environ, {"STRATA_TOKEN_GAP_RAW": "0"}):
            receipt = g.receipt()
        self.assertNotIn("token_gaps_ms", receipt)
        merged = request_stats([receipt, receipt])
        self.assertFalse(any(k.startswith("token_gap") for k in merged))


class DirectSlotProtocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.script = Path(self.tmp.name) / "fake_slot_only.py"
        self.script.write_text(FAKE_SLOT_ONLY, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        self.engine = self.start(["--batch", "2", "--slotcache", "--batch-slot-only", "--log", str(self.log)])

    def start(self, args):
        popen = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen", lambda cmd, **kw: popen(
                [sys.executable, str(self.script), *cmd[1:]], **kw)):
            return StrataEngine("strata", args)

    def tearDown(self):
        self.engine.unload()
        self.tmp.cleanup()

    def run_request(self, text, max_new):
        out = [t for t in self.engine.generate(list(text.encode()), max_new, {}, threading.Event()) if t is not None]
        return out, dict(self.engine.last)

    def assert_clock(self, out, last):
        self.assertEqual(last["token_gap_tokens"], len(out))
        self.assertEqual(last["token_gap_count"], max(0, len(out) - 1))
        self.assertEqual(last["token_gap_source"], "engine_stdout_token_id")

    def test_first_request_is_direct_and_one_token_finishes(self):
        out, last = self.run_request("hello", 1)
        self.assertEqual(len(out), 1)
        self.assert_clock(out, last)
        self.assertTrue(self.log.read_text().startswith("BGEN 0 "))

    def test_singleton_stays_in_slot1(self):
        self.engine.slot_busy[0] = True
        try:
            out, last = self.run_request("LONGREPLY hello", 45)
        finally:
            self.engine.slot_busy[0] = False
        self.assertEqual(len(out), 45)
        self.assert_clock(out, last)
        self.assertFalse(self.engine._may_go_solo(64, 0, None))
        self.assertEqual([x.split()[0:2] for x in self.log.read_text().splitlines()], [["BGEN", "1"]])

    def test_four_queued_requests_have_private_clocks(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda i: self.run_request(f"hello {i}", 64), range(4)))
        for out, last in results:
            self.assertEqual(out, list(b"ok, done.") + [257])
            self.assert_clock(out, last)
        commands = [line.split() for line in self.log.read_text().splitlines()]
        self.assertEqual(len(commands), 4)
        self.assertTrue(all(row[0] == "BGEN" for row in commands))
        self.assertTrue(all(row[1] in ("0", "1") for row in commands))

    def test_service_completion_waits_for_native_done_and_keeps_clock(self):
        svc = Service(self.engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        for limit in (64, 4):
            with mock.patch.dict(os.environ, {"STRATA_TOKEN_GAP_RAW": "1"}), mock.patch.object(server, "EOS_IDS", {257}):
                events = list(svc.run(list(b"helloX"), False, [], limit, {}, threading.Event()))
            done = next(value for kind, value in events if kind == "done")
            self.assertEqual(done["completion_tokens"], min(limit, 10))
            self.assertFalse(any(self.engine.slot_busy))
            self.assertEqual(self.engine.last["generated"], done["completion_tokens"])
            timings = done["timings"]
            self.assertEqual(timings["token_gap_tokens"], done["completion_tokens"])
            self.assertEqual(len(timings["token_gaps_ms"]), done["completion_tokens"] - 1)
            self.assertGreaterEqual(timings["token_gap_last_ns"], timings["token_gap_first_ns"])

    def test_completed_output_nonfinite_releases_all_admission_claims(self):
        self.engine.unload()
        fake = FAKE_SLOT_ONLY.replace(
            "fin = 'stop' if t == 257 else 'cancel' if b in stopped else 'length'",
            "fin = 'nonfinite' if 'nf-end' not in fired else 'length'; fired.add('nf-end')")
        self.script.write_text(fake, encoding="utf-8")
        self.engine = self.start(["--batch", "2", "--slotcache", "--batch-slot-only"])
        stream = self.engine.generate(list(b"hello"), 3, {}, threading.Event())
        out = []
        with self.assertRaises(server.EngineNonFinite):
            for token in stream:
                if token is not None: out.append(token)
        self.assertEqual(len(out), 3)
        self.assertFalse(any(self.engine.slot_busy))
        self.assertFalse(self.engine.slot_claims)
        self.assertTrue(all(not held for held in self.engine.slot_held))
        self.assertTrue(self.engine.ctl.acquire(blocking=False))
        self.engine.ctl.release()
        out, last = self.run_request("next healthy request", 3)
        self.assertEqual(len(out), 3)
        self.assert_clock(out, last)

    def test_pending_drain_is_not_reported_idle(self):
        self.engine.slot_busy[1] = True
        self.assertEqual(self.engine.slots_view()[1]["state"], "draining")
        self.engine.slot_busy[1] = False
        self.assertEqual(self.engine.slots_view()[1]["state"], "idle")

    def test_missing_capability_fails_closed(self):
        self.script.write_text(FAKE_BATCH, encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "capability handshake mismatch"):
            self.start(["--batch", "2", "--batch-slot-only"])


if __name__ == "__main__":
    unittest.main()
