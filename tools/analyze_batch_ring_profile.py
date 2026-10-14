#!/usr/bin/env python3
"""Close host ring/dispatch/native pool bins. No GPU useful-compute attribution."""
import argparse
import json
from pathlib import Path
from analyze_batch_host_profile import KV, analyze as analyze_host, stats

RING = ('wait_ns', 'callback_ns', 'publish_ns', 'other_ns')
DISPATCH = ('plan_ns', 'actq_ns', 'jobs_ns', 'run_ns', 'dispatch_other_ns')
PHASE = ('park_before_ns', 'phase_publish_ns', 'host_work_ns', 'wait_done_ns', 'park_after_ns', 'phase_other_ns')
REQUIRED = ('id', 'rows', 'steps', 'ring_ns', 'dispatch_calls', 'dispatch_ns', 'callback_other_ns',
            'zero_cpu_calls', 'cpu_jobs', 'cpu_needed_tokens', 'quantized_tokens', 'phase_calls', 'phase_ns',
            'native_calls', 'native_ns', 'native_phase_ns', 'ff_quant_ns', 'native_other_ns', 'run_other_ns') + RING + DISPATCH + PHASE


def validate(r, h, round_):
    if set(REQUIRED) - r.keys(): raise ValueError('missing ring fields')
    if any(r[k] < 0 for k in REQUIRED): raise ValueError('negative ring bin/count')
    if r['rows'] != h['rows'] or r['id'] != h['id']: raise ValueError('ring id/rows mismatch')
    if r['ring_ns'] != h['ring_ns']: raise ValueError('ring/host span mismatch')
    if r['steps'] != 48 * (1 if round_['merged'] else int(round_['dense_groups'])):
        raise ValueError('unexpected frozen-model ring step count')
    equations = (
        r['ring_ns'] == sum(r[k] for k in RING),
        r['callback_ns'] == r['dispatch_ns'] + r['callback_other_ns'],
        r['dispatch_ns'] == sum(r[k] for k in DISPATCH),
        r['run_ns'] == r['native_ns'] + r['run_other_ns'],
        r['native_ns'] == r['native_phase_ns'] + r['ff_quant_ns'] + r['native_other_ns'],
        r['phase_ns'] == sum(r[k] for k in PHASE),
        r['native_phase_ns'] == r['phase_ns'],
        r['dispatch_calls'] == r['steps'],
        r['zero_cpu_calls'] + r['native_calls'] == r['dispatch_calls'],
        r['phase_calls'] >= 2*r['native_calls'], r['phase_calls'] % 2 == 0,
        r['cpu_jobs'] >= r['native_calls'],
        r['cpu_needed_tokens'] <= r['quantized_tokens'] <= r['rows'] * r['dispatch_calls'],
    )
    if not all(equations): raise ValueError(f'ring closure/counters failed id={r["id"]}')


def analyze(path):
    host = analyze_host(path)
    rings, hosts, rounds, samples = {}, {}, {}, {}
    with Path(path).open(errors='replace') as handle:
        for line in handle:
            prefix = line.split(' ', 1)[0]
            if prefix not in ('batch-ring', 'batch-host', 'batch-round', 'batch-ring-step'): continue
            data = {k: (float(v) if prefix == 'batch-round' and '.' in v else int(v)) for k,v in KV.findall(line.split(' slot=',1)[0])}
            ident = data['id']
            if prefix == 'batch-ring-step':
                steps = samples.setdefault(ident, {})
                if data['step'] in steps: raise ValueError('duplicate ring sample step')
                steps[data['step']] = data
            else:
                dest = {'batch-ring': rings, 'batch-host': hosts, 'batch-round': rounds}[prefix]
                if ident in dest: raise ValueError('duplicate ring/host/round id')
                dest[ident] = data
    if not rings or rings.keys() != hosts.keys() or rings.keys() != rounds.keys():
        raise ValueError('ring/host/round coverage incomplete')
    for ident, r in rings.items():
        validate(r, hosts[ident], rounds[ident])
        expected = (ident & 15) == 0
        if expected != (ident in samples): raise ValueError('sample coverage differs from stride16 contract')
        if not expected: continue
        values = samples[ident]
        if sorted(values) != list(range(r['steps'])): raise ValueError('sample step coverage incomplete')
        for step, s in values.items():
            if any(s[k] < 0 for k in ('layer','group','wait_ns','callback_ns','publish_ns') + DISPATCH[:4]):
                raise ValueError('negative sampled step')
            groups = r['steps']//48
            if s['layer'] != step//groups or s['group'] != step%groups: raise ValueError('sample layer/group order')
        for key in ('wait_ns','callback_ns','publish_ns') + DISPATCH[:4]:
            if sum(s[key] for s in values.values()) != r[key]: raise ValueError('sample aggregate does not close')
    if set(samples) - rings.keys(): raise ValueError('orphan sampled steps')
    bins = ('ring_ns', 'callback_other_ns', 'dispatch_ns', 'native_ns', 'ff_quant_ns', 'native_other_ns', 'run_other_ns', 'phase_ns') + RING + DISPATCH + PHASE
    quantized = sum(r['quantized_tokens'] for r in rings.values())
    needed = sum(r['cpu_needed_tokens'] for r in rings.values())
    return {'log': str(path), 'windows': len(rings), 'sampled_windows': len(samples), 'host_closure_pass':host['closure_pass'],
            'ring_closure_pass':True, 'scope':'nested HOST wall bins; not pure GPU time; diagnostic-only',
            'phase_stats':{k:stats([r[k] for r in rings.values()]) for k in bins},
            'counts':{k:sum(r[k] for r in rings.values()) for k in ('steps','dispatch_calls','zero_cpu_calls','cpu_jobs','phase_calls','native_calls','cpu_needed_tokens','quantized_tokens')},
            'redundant_quantized_tokens':quantized-needed,
            'CPU_needed_fraction_of_quantized':needed/quantized if quantized else None}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path); parser.add_argument('--out',type=Path)
    args=parser.parse_args(); text=json.dumps(analyze(args.log),indent=2)+'\n'
    if args.out: args.out.write_text(text)
    else: print(text,end='')
