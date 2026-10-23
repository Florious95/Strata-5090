"""CPU regressions for upstream telemetry/summary plus gold wide-window accounting."""
import unittest
from serve import server


class GoldMerge(unittest.TestCase):
    def test_wide_window_telemetry_is_work_not_emitted_tokens(self):
        for rows in (8, 9, 12, 16):
            sample = server.parse_batch_window_sample(f"BSTAT 1 {rows} 2 20\n")
            self.assertEqual(sample["processed_rows"], rows)
            self.assertEqual(sample["active_rows"], 2)
        self.assertIsNone(server.parse_batch_window_sample("BSTAT 1 17 2 20\n"))

    def test_summary_keeps_first_chunk_and_merges_native_token_gaps(self):
        segments = [dict(chunk=256, reused=29, prompt_read=10, generated=2,
                         token_gap_tokens=2, token_gaps_ms=[2.0],
                         token_gap_first_ns=1_000_000, token_gap_last_ns=3_000_000),
                    dict(chunk=512, reused=31, prompt_read=0, generated=2,
                         token_gap_tokens=2, token_gaps_ms=[3.0],
                         token_gap_first_ns=7_000_000, token_gap_last_ns=10_000_000)]
        stats = server.request_stats(segments)
        self.assertEqual((stats["chunk"], stats["reused"], stats["prompt_read"]), (256, 29, 10))
        self.assertEqual(stats["token_gaps_ms"], [2.0, 4.0, 3.0])
        self.assertEqual(stats["token_gap_tokens"], 4)
        self.assertEqual(stats["token_gap_p95_ms"], 4.0)

    def test_summary_does_not_mislabel_partial_latency_percentiles(self):
        stats = server.request_stats([dict(chunk=256), dict(chunk=512, token_gap_p95_ms=7)])
        self.assertNotIn("token_gap_p95_ms", stats)
        self.assertEqual(stats["chunk"], 256)


if __name__ == "__main__":
    unittest.main()
