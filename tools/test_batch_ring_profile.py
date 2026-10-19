import copy
from pathlib import Path
import tempfile
import unittest
from analyze_batch_ring_profile import REQUIRED, analyze, validate
from test_batch_host_profile import fixture as host_fixture, lines as host_lines


def fixture(ident=16):
    h, round_ = host_fixture(ident)
    round_.update(merged=1, dense_groups=2)
    r = {k:0 for k in REQUIRED}
    r.update(id=ident,rows=11,steps=48,ring_ns=h['ring_ns'],wait_ns=48000,callback_ns=240000,
             publish_ns=48000,other_ns=164000,dispatch_calls=48,dispatch_ns=216000,
             plan_ns=96000,actq_ns=24000,jobs_ns=24000,run_ns=57600,dispatch_other_ns=14400,
             callback_other_ns=24000,cpu_jobs=48,cpu_needed_tokens=48,quantized_tokens=528,
             phase_calls=96,phase_ns=38400,park_before_ns=3840,phase_publish_ns=1920,
             host_work_ns=19200,wait_done_ns=7680,park_after_ns=3840,phase_other_ns=1920,
             native_calls=48,native_ns=48000,native_phase_ns=38400,ff_quant_ns=4800,
             native_other_ns=4800,run_other_ns=9600)
    steps=[dict(id=ident,step=i,layer=i,group=0,wait_ns=1000,callback_ns=5000,publish_ns=1000,
                plan_ns=2000,actq_ns=500,jobs_ns=500,run_ns=1200) for i in range(48)]
    return h,round_,r,steps


def lines(h, round_, r, steps):
    text=host_lines(h,round_)+'batch-ring '+' '.join(f'{k}={v}' for k,v in r.items())+'\n'
    for s in steps:
        text+='batch-ring-step '+' '.join(f'{k}={v}' for k,v in s.items())+'\n'
    return text


class RingAccounting(unittest.TestCase):
    def test_valid_native_and_layer_samples(self):
        h,q,r,s=fixture(); validate(r,h,q)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'log'; path.write_text(lines(h,q,r,s)); out=analyze(path)
        self.assertTrue(out['ring_closure_pass'])
        self.assertEqual(out['redundant_quantized_tokens'],480)
        self.assertEqual(out['sampled_windows'],1)

    def test_each_required_missing_or_negative(self):
        h,q,r,_=fixture()
        for k in REQUIRED:
            bad=copy.copy(r); del bad[k]
            with self.assertRaises(ValueError): validate(bad,h,q)
            bad=copy.copy(r); bad[k]=-1
            with self.assertRaises(ValueError): validate(bad,h,q)

    def test_closures_and_counters_are_powered(self):
        h,q,r,_=fixture()
        fields=['ring_ns','wait_ns','callback_ns','publish_ns','other_ns','dispatch_ns','plan_ns',
                'actq_ns','jobs_ns','run_ns','dispatch_other_ns','callback_other_ns','phase_ns',
                'park_before_ns','phase_publish_ns','host_work_ns','wait_done_ns','park_after_ns',
                'phase_other_ns','native_ns','native_phase_ns','ff_quant_ns','native_other_ns',
                'run_other_ns','steps','dispatch_calls','zero_cpu_calls','native_calls','phase_calls','quantized_tokens']
        for k in fields:
            bad=copy.copy(r); bad[k]+=1
            with self.assertRaises(ValueError,msg=k): validate(bad,h,q)

    def test_zero_cpu_layers_are_valid(self):
        h,q,r,_=fixture()
        for k in ['cpu_jobs','cpu_needed_tokens','quantized_tokens','phase_calls','phase_ns','park_before_ns',
                  'phase_publish_ns','host_work_ns','wait_done_ns','park_after_ns','phase_other_ns',
                  'native_calls','native_ns','native_phase_ns','ff_quant_ns','native_other_ns']: r[k]=0
        r['zero_cpu_calls']=48; r['run_other_ns']=r['run_ns']; validate(r,h,q)

    def test_coverage_duplicate_order_and_sum_fail_closed(self):
        h,q,r,s=fixture(); good=lines(h,q,r,s)
        wrong=copy.deepcopy(s); wrong[4]['layer']=3
        changed=copy.deepcopy(s); changed[4]['wait_ns']+=1
        cases=[host_lines(h,q),good+good,lines(h,q,r,s[:-1]),lines(h,q,r,s+[s[0]]),
               lines(h,q,r,wrong),lines(h,q,r,changed)]
        h1,q1,r1,s1=fixture(1); cases.append(lines(h1,q1,r1,s1))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'log'
            for text in cases:
                path.write_text(text)
                with self.assertRaises(ValueError): analyze(path)

    def test_unsampled_native_window_valid(self):
        h,q,r,_=fixture(1)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'log'; path.write_text(lines(h,q,r,[])); out=analyze(path)
        self.assertEqual(out['windows'],1)
        self.assertEqual(out['sampled_windows'],0)


if __name__=='__main__': unittest.main()
