"""Stable logical request affinity, not a token-prefix guess. No GPU or private benchmark tests."""
import hashlib
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

import server


def engine(slots=8):
    e = object.__new__(server.StrataEngine)
    e.batch = slots
    e.info = {"batch_generalized": 1, "batch_window_audit": 1}
    e.slot_order = list(range(slots))
    e.slot_busy = [False] * slots
    e.slot_used = [0] * slots
    e.slot_group = [0] * slots
    e.slot_groups, e.slot_gs = 1, 0
    e.slot_held = [[] for _ in range(slots)]
    e.slot_keys, e.slot_request_keys = [None] * slots, [None] * slots
    return e


class CacheKey(unittest.TestCase):
    def test_long_shared_prefix_independent_keys_fill_all_eight_and_follow_their_owners(self):
        e, owners = engine(), {}
        prefix = list(range(51592))
        prompts = [prefix + [1000000 + i, 7, 8] for i in range(8)]
        for lane, ids in enumerate(prompts):
            key = f"lane-{lane}"
            b = e.pick_slot(ids, key)
            self.assertNotIn(b, owners.values())
            e.slot_held[b], e.slot_keys[b] = list(ids), key
            owners[lane] = b
        self.assertEqual(len(set(owners.values())), 8)
        for lane in reversed(range(8)):
            # Branch/re-render the reply tail: a key gives owner affinity, NOT permission to bypass native LCP.
            self.assertEqual(e.pick_slot(prompts[lane][:-1] + [999], f"lane-{lane}"), owners[lane])

    def test_bound_rewritten_tail_beats_an_empty_slot(self):
        e = engine(2)
        e.slot_held[0], e.slot_keys[0] = list(range(30000)), "same-conversation"
        self.assertEqual(e.pick_slot(list(range(29000)) + [-1], "same-conversation"), 0)
        self.assertEqual(e.pick_slot(list(range(29000)) + [-1], "independent-conversation"), 1)

    def test_busy_and_provisional_bound_key_never_duplicates_on_a_free_owner(self):
        e = engine(2)
        e.slot_busy[0] = True
        e.slot_request_keys[0] = "reading"
        self.assertIsNone(e.pick_slot([1, 2], "reading"))
        e.slot_held[0], e.slot_keys[0] = [1], "decoding"
        self.assertIsNone(e.pick_slot([1, 2], "decoding"))
        self.assertEqual(e.pick_slot([1, 2], "independent"), 1)

    def test_cleared_physical_state_invalidates_binding_before_lookup(self):
        e = engine(2)
        e.slot_keys[0], e.slot_request_keys[0] = "gone", "stale-reservation"
        self.assertEqual(e.pick_slot([1], "new"), 0)
        self.assertIsNone(e.slot_keys[0])
        self.assertIsNone(e.slot_request_keys[0])
        e.slot_held[0], e.slot_keys[0] = [1, 2], "retained"
        e.pick_slot([9], "other")
        self.assertEqual(e.slot_keys[0], "retained")  # busy flag release alone does not destroy warm affinity

    def test_without_a_key_the_original_long_lcp_policy_is_unchanged(self):
        e = engine(2)
        e.slot_held[0] = list(range(51592)) + [-1]
        self.assertEqual(e.pick_slot(list(range(51592)) + [7]), 0)

    def test_prepare_validates_hashes_and_retains_key_across_internal_continuations(self):
        svc = object.__new__(server.Service)
        svc.vision, svc._prepare = None, Mock(return_value=([1], False, 1))
        svc.engine = engine()
        cancel = threading.Event()
        ids = svc.prepare([], [], {}, req={"prompt_cache_key": "lane-4"}, cancel=cancel)
        self.assertEqual(ids, ([1], False, 1))
        expected = hashlib.sha256(b"lane-4").hexdigest()
        self.assertEqual(cancel._strata_prompt_cache_key, expected)
        svc.prepare([], [], {}, cancel=cancel)
        self.assertEqual(cancel._strata_prompt_cache_key, expected)
        for value in ("", "x" * 65, 1, True, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                svc.prepare([], [], {}, req={"prompt_cache_key": value}, cancel=threading.Event())
        fresh = threading.Event()
        svc.prepare([], [], {}, req={"prompt_cache_key": None}, cancel=fresh)
        self.assertFalse(hasattr(fresh, "_strata_prompt_cache_key"))

    def test_default_service_preserves_upstream_cache_group_semantics(self):
        svc = object.__new__(server.Service)
        svc.vision, svc._prepare = None, Mock(return_value=([1], False, 1))
        svc.engine = engine()
        svc.engine.info = {}
        for value in ("shared-parent-subagent-group", "", "x" * 65, 1):
            with self.subTest(value=value):
                cancel = threading.Event()
                svc.prepare([], [], {}, req={"prompt_cache_key": value}, cancel=cancel)
                self.assertFalse(hasattr(cancel, "_strata_prompt_cache_key"))

    @unittest.skipIf(sys.platform == "win32", "POSIX fake executable fixture")
    def test_key_reaches_native_bgen_owner_and_reuses_it_instead_of_solo(self):
        # Existing public fake protocol, not a bench acceptance implementation.
        from test_parallel import FAKE_BATCH
        with tempfile.TemporaryDirectory() as directory:
            script, log = Path(directory) / "fake-engine", Path(directory) / "protocol.log"
            script.write_text(f'#!{sys.executable}\nprint("INFO batch_window_audit=1", flush=True)\n' + FAKE_BATCH)
            script.chmod(0o700)
            e = server.StrataEngine(str(script), ["--batch", "4", "--slotcache", "--log", str(log)],
                                    env={"STRATA_BATCH_GENERALIZED": "0"})
            try:
                for key in ("one", "two", "one"):
                    cancel = threading.Event()
                    cancel._strata_prompt_cache_key = key
                    self.assertEqual(len(list(e.generate([1, 2, 3], 2, {}, cancel))), 2)
                sent = [line.split()[:2] for line in log.read_text().splitlines() if line.startswith(("GEN ", "BGEN "))]
                self.assertEqual(sent, [["BGEN", "0"], ["BGEN", "1"], ["BGEN", "0"]])
                self.assertEqual(e.slot_keys[:2], ["one", "two"])
            finally:
                e.close()


if __name__ == "__main__":
    unittest.main()
