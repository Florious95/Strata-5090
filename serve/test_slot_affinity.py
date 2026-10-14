"""CPU routing contract: LCP chooses an idle slot, never bypasses native checkpoint checks."""
import unittest
from serve.server import StrataEngine


def engine(held, used=None, busy=None, groups=1):
    e = StrataEngine.__new__(StrataEngine)
    e.batch, e.slot_held = len(held), held
    e.slot_order = list(range(e.batch))
    e.slot_busy = busy or [False] * e.batch
    e.slot_used = used or list(range(e.batch))
    e.slot_groups = groups
    e.slot_gs = 0
    e.slot_group = [i % groups for i in range(e.batch)]
    return e


class Affinity(unittest.TestCase):
    def test_event_a_shorter_rerender_keeps_hot_slot1(self):
        prompt = list(range(49252))
        old0 = prompt[:23575]; old0[16515] = -1
        hot1 = list(range(50266)); hot1[45172] = -2  # prior reply's thinking is rewritten
        e = engine([old0, hot1], used=[1, 2])
        self.assertEqual(e.pick_slot(prompt), 1)
    def test_event_b_rewritten_reply_keeps_hot_slot0(self):
        prompt = list(range(73276))
        hot0 = prompt[:73238]; hot0[68958] = -1
        old1 = prompt[:50266]; old1[41452] = -2
        e = engine([hot0, old1], used=[2, 1])
        self.assertEqual(e.pick_slot(prompt), 0)
    def test_exact_full_prefix_still_wins(self):
        self.assertEqual(engine([[1, 9], [1, 2]]).pick_slot([1, 2, 3]), 1)
    def test_longer_held_state_can_affiliate_a_shorter_prompt(self):
        self.assertEqual(engine([[1, 9], [1, 2, 3, 4]]).pick_slot([1, 2, 3]), 1)
    def test_busy_hot_slot_is_never_selected_or_waited_for(self):
        self.assertEqual(engine([[1, 9], [1, 2, 3]], busy=[False, True]).pick_slot([1, 2, 3, 4]), 0)
    def test_all_busy_returns_none(self):
        self.assertIsNone(engine([[1], [1]], busy=[True, True]).pick_slot([1, 2]))
    def test_no_shared_prefix_prefers_empty_then_lru(self):
        self.assertEqual(engine([[8, 9], []], used=[0, 4]).pick_slot([1, 2]), 1)
        self.assertEqual(engine([[8], [9]], used=[2, 1]).pick_slot([1, 2]), 1)
    def test_ties_preserve_slot_order(self):
        e = engine([[1, 2, 8], [1, 2, 9]])
        e.slot_order = [1, 0]
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)
    def test_group_threshold_and_pipeline_hole_policy_unchanged(self):
        e = engine([[1, 2], []], groups=2)
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)  # <512: keep the existing group-balancing fallback
        e = engine([[1, 2], []], busy=[False, True])
        e.slot_gs = 2
        self.assertEqual(e.pick_slot([1, 2, 3]), 0)
    def test_does_not_mutate_prompt_or_held_states(self):
        h = [[1, 2, 8], [1, 2, 9]]; p = [1, 2, 3]
        engine(h).pick_slot(p)
        self.assertEqual(h, [[1, 2, 8], [1, 2, 9]])
        self.assertEqual(p, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
