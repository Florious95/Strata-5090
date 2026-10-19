import copy
from pathlib import Path
import tempfile
import unittest

from analyze_batch_host_profile import REQUIRED, RUN, analyze, validate


def fixture(ident=1):
    h = {key: 0 for key in REQUIRED}
    h.update(id=ident, rows=11, stage_ns=20000, stage_data_ns=10000, ple_gather_ns=9000,
             stage_other_ns=10000, ring_ns=500000, tail_sync_ns=200000,
             context_enter_ns=100, context_exit_ns=50, launch_ns=1500, query_ns=500,
             collect_ns=100, sample_ns=1000, out_ns=500, progress_ns=1000,
             run_other_ns=10000, return_ns=2000, admin_ns=5000,
             graph_cache_hit=1, end_mono_ns=9223372036854770000 + ident)
    h['run_ns'] = h['run_other_ns'] + sum(h[k] for k in RUN)
    h['call_ns'] = h['run_ns'] + h['return_ns']
    h['verify_ns'] = h['admin_ns'] + h['call_ns']
    r = dict(id=ident, rows=11, active=2, head_type=13, head_wide_active=1,
             verify_ms=h['verify_ns']/1000000)
    return h, r


def lines(h, r):
    # Real trace rounds to three decimal ms and repeats rows in slot fragments.
    return ('batch-round ' + ' '.join(f'{k}={v:.3f}' if k == 'verify_ms' else f'{k}={v}'
            for k, v in r.items()) + ' slot=0:rows=6 slot=1:rows=5\n' +
            'batch-host ' + ' '.join(f'{k}={v}' for k, v in h.items()) + '\n')


class Accounting(unittest.TestCase):
    def test_valid_rounding_nested_bins_and_exact_64bit(self):
        h, r = fixture()
        validate(h, r)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'engine.log'
            path.write_text(lines(h, r))
            got = analyze(path)
            self.assertTrue(got['closure_pass'])
            self.assertEqual(got['top_windows'][0]['rows'], 11)
            self.assertEqual(got['top_windows'][0]['end_mono_ns'], h['end_mono_ns'])

    def test_every_required_field_fails_closed_when_missing(self):
        h, r = fixture()
        for key in REQUIRED:
            bad = copy.copy(h); del bad[key]
            with self.assertRaises(ValueError): validate(bad, r)

    def test_negative_each_bin_fails(self):
        h, r = fixture()
        for key in REQUIRED:
            bad = copy.copy(h); bad[key] = -1
            with self.assertRaises(ValueError): validate(bad, r)

    def test_every_closure_and_ple_subset_has_power(self):
        h, r = fixture()
        for key in ['verify_ns', 'call_ns', 'run_ns', 'stage_ns', 'run_other_ns', 'stage_other_ns']:
            bad = copy.copy(h); bad[key] += 1
            with self.assertRaises(ValueError): validate(bad, r)
        bad = copy.copy(h); bad['ple_gather_ns'] = bad['stage_data_ns']+1
        with self.assertRaises(ValueError): validate(bad, r)

    def test_coverage_duplicate_and_nonmonotonic_fail(self):
        h, r = fixture(); h2, r2 = fixture(2)
        h2['end_mono_ns'] = h['end_mono_ns']-1
        cases = ['batch-round id=1 rows=11 verify_ms=1\n', lines(h, r)*2,
                 lines(h, r)+lines(h2, r2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'engine.log'
            for text in cases:
                path.write_text(text)
                with self.assertRaises(ValueError): analyze(path)


if __name__ == '__main__':
    unittest.main()
