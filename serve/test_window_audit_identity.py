"""Native audit identity survives HTTP formatting; no engine/GPU is started."""
import threading
import unittest
from unittest.mock import Mock

import responses
import server


def done(slot=5, generation=9):
    suffix = "" if slot is None else f" slot_id={slot} request_generation={generation}"
    return "DONE 1 258000 120.0 10.0 length 0 0 257922 456 480 0 0 0.0 78 0" + suffix


class WindowAuditIdentityTests(unittest.TestCase):
    def engine(self, info=None):
        engine = object.__new__(server.StrataEngine)
        engine.batch = 8
        engine.info = info or {}
        return engine

    def test_parse_done_uses_native_identity_not_reserved_server_slot(self):
        engine = self.engine()
        engine._parse_done(done())
        timing = server.request_timings(258000, 1, engine.last)
        self.assertEqual((timing["slot_id"], timing["request_generation"]), (5, 9))
        self.assertEqual(timing["slot_generations"], [{"slot_id": 5, "request_generation": 9}])

    def test_legacy_done_does_not_invent_generation_zero(self):
        engine = self.engine()
        engine._parse_done(done(None))
        timing = server.request_timings(258000, 1, engine.last)
        self.assertNotIn("slot_id", timing)
        self.assertNotIn("request_generation", timing)
        self.assertNotIn("slot_generations", timing)

    def test_partial_or_invalid_native_identity_is_not_forwarded(self):
        for suffix in (" slot_id=1", " request_generation=2", " slot_id=-1 request_generation=2",
                       " slot_id=1 request_generation=0", " slot_id=1 request_generation=oops"):
            engine = self.engine()
            engine._parse_done(done(None) + suffix)
            self.assertFalse({"slot_id", "request_generation", "slot_generations"} & engine.last.keys())

    def test_empty_slot_seeding_preserves_meaningful_lcp_and_default_policy(self):
        prompt = list(range(30001))
        engine = self.engine({"batch_generalized": "1"})
        engine.batch = 4
        engine.slot_order = list(range(4))
        engine.slot_busy = [False] * 4
        engine.slot_group = [0] * 4
        engine.slot_used = [0] * 4
        engine.slot_held = [prompt[:12], [], [], []]
        self.assertEqual(engine.pick_slot(prompt), 1)
        engine.slot_held[0] = prompt[:29000] + [-1]
        self.assertEqual(engine.pick_slot(prompt), 0)
        engine.slot_held = [prompt[:12], [], [], []]
        engine.info = {"batch_window_audit": "1"}  # reference arm uses the same policy
        self.assertEqual(engine.pick_slot(prompt), 1)
        engine.info = {}
        self.assertEqual(engine.pick_slot(prompt), 0)
        engine.info = {"batch_generalized": "1"}
        engine.slot_held = [prompt[:12], [-2], [-3], [-4]]  # no empty owners: keep even a short useful prefix
        self.assertEqual(engine.pick_slot(prompt), 0)

    def test_parallel_last_identity_is_thread_local(self):
        engine, barrier, results = self.engine(), threading.Barrier(2), []
        def parse(slot):
            engine._parse_done(done(slot, slot + 1))
            barrier.wait(timeout=5)
            results.append(server.request_timings(258000, 1, engine.last)["slot_generations"])
        threads = [threading.Thread(target=parse, args=(slot,)) for slot in (2, 7)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertCountEqual(results, [[{"slot_id": 2, "request_generation": 3}],
                                        [{"slot_id": 7, "request_generation": 8}]])

    def test_continuations_keep_all_native_legs(self):
        legs = [{"slot_id": 1, "request_generation": 2}, {"slot_id": 7, "request_generation": 3}]
        segments = [{"slot_generations": [legs[0]], "generated": 1},
                    {"slot_generations": legs, "generated": 2}]
        result = server.request_stats(segments)
        self.assertEqual(result["slot_generations"], legs)
        self.assertEqual((result["slot_id"], result["request_generation"]), (7, 3))
        self.assertEqual(result["generated"], 3)

    def test_canonical_and_audited_reference_never_take_unowned_solo_leg(self):
        for info in ({"canonical_slots": "1"}, {"slot_only": "1"}, {"batch_window_audit": "1"}):
            self.assertTrue(self.engine(info)._requires_slot_admission())
        self.assertFalse(self.engine({"canonical_slots": "0", "batch_window_audit": "0"})._requires_slot_admission())

    def test_anthropic_stream_and_collect_keep_audit_timings(self):
        timing = {"slot_id": 5, "request_generation": 9,
                  "slot_generations": [{"slot_id": 5, "request_generation": 9}]}
        svc = Mock()
        svc.model_for.return_value = "synthetic"
        svc.run.return_value = iter([("done", {"finish": "stop", "completion_tokens": 1, "timings": timing})])
        events = list(server.anthropic_events(svc, {}, [1], False, [], 1, threading.Event()))
        self.assertEqual(next(data["timings"] for name, data in events if name == "message_delta"), timing)
        self.assertEqual(server.anthropic_collect(events)["timings"], timing)

    def test_responses_complete_and_incomplete_keep_audit_timings(self):
        timing = {"slot_id": 5, "request_generation": 9,
                  "slot_generations": [{"slot_id": 5, "request_generation": 9}]}
        for finish in ("stop", "length"):
            asm = responses.Assembler({}, "synthetic", 1, {}, False)
            events = asm.finish({"finish": finish, "completion_tokens": 1, "timings": timing})
            self.assertEqual(events[-1]["response"]["timings"], timing)

    def test_response_without_native_audit_identity_keeps_original_shape(self):
        asm = responses.Assembler({}, "synthetic", 1, {}, False)
        event = asm.finish({"finish": "stop", "completion_tokens": 1, "timings": {"predicted_n": 1}})[-1]
        self.assertNotIn("timings", event["response"])


if __name__ == "__main__":
    unittest.main()
