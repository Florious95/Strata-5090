import tempfile
import unittest
from pathlib import Path
from analyze_batch_qsa_tile import analyze,validate


def row(**kw):
    d=dict(rows=12,qsa_tile_active=1,qsa_tile_calls=24,qsa_tile_rows=144,qsa_scalar_rows=0);d.update(kw);return d


class TestQsaTile(unittest.TestCase):
    def test_on_off_partial(self):
        for d in (row(),row(qsa_tile_active=0,qsa_tile_calls=0,qsa_tile_rows=0,qsa_scalar_rows=144),
                  row(qsa_tile_calls=12,qsa_tile_rows=72,qsa_scalar_rows=72)):
            validate(d)
    def test_bad_counters(self):
        for d in (row(qsa_tile_calls=-1),row(qsa_tile_active=0),row(qsa_tile_rows=143),row(qsa_scalar_rows=12),
                  row(qsa_tile_calls=1),row(qsa_tile_rows=12,qsa_scalar_rows=132),row(qsa_tile_rows=144.0)):
            with self.assertRaises(ValueError):validate(d)
        d=row();del d['qsa_tile_active']
        with self.assertRaises(ValueError):validate(d)
    def test_parse_duplicates_and_scope(self):
        line='batch-round id=1 '+' '.join(f'{k}={v}' for k,v in row().items())+' slot=0:rows=6\n'
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'engine.log';p.write_text(line)
            d=analyze(p);self.assertEqual(d['removed_selection_attention_kernel_nodes'],480)
            p.write_text(line+line)
            with self.assertRaises(ValueError):analyze(p)
            p.write_text('warmup only\n')
            with self.assertRaises(ValueError):analyze(p)


if __name__=='__main__':unittest.main()
