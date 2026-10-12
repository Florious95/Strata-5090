#!/usr/bin/env python3
"""Join batch-host/batch-round by native window id; fail closed on coverage/bin errors.

Only timing/dispatch metadata is read. No output text, prompts or hidden needles.
Host wall-time is not GPU kernel time; nested stage/PLE bins are not additive.
"""
import argparse
import json
import math
from pathlib import Path
import re

KV = re.compile(r'\b([a-z_]+)=(-?\d+(?:\.\d+)?)')
RUN = ('context_enter_ns', 'context_exit_ns', 'stage_ns', 'launch_ns', 'query_ns',
       'ring_ns', 'tail_sync_ns', 'copy_sync_ns', 'collect_ns', 'sample_ns', 'out_ns', 'progress_ns')
STAGE = ('evict_ns', 'capture_exec_ns', 'capture_commit_ns', 'stage_data_ns')
REQUIRED = ('verify_ns', 'admin_ns', 'call_ns', 'return_ns', 'run_ns', 'run_other_ns',
            'stage_other_ns', 'ple_gather_ns', 'end_mono_ns', 'graph_cache_hit') + RUN + STAGE


def validate(h, r):
    missing = sorted(set(REQUIRED + ('id', 'rows')) - h.keys())
    if missing:
        raise ValueError(f'missing host fields: {missing}')
    if h['id'] != r['id'] or h['rows'] != r['rows']:
        raise ValueError('window id/rows mismatch')
    if any(h[k] < 0 for k in REQUIRED):
        raise ValueError('negative host time/counter')
    if h['graph_cache_hit'] not in (0, 1):
        raise ValueError('invalid graph cache hit')
    equations = (
        h['verify_ns'] == h['admin_ns'] + h['call_ns'],
        h['call_ns'] == h['run_ns'] + h['return_ns'],
        h['run_ns'] == h['run_other_ns'] + sum(h[k] for k in RUN),
        h['stage_ns'] == h['stage_other_ns'] + sum(h[k] for k in STAGE),
        h['ple_gather_ns'] <= h['stage_data_ns'],
        abs(h['verify_ns'] - round(r['verify_ms'] * 1000000)) <= 1001,
    )
    if not all(equations):
        raise ValueError(f'host bins fail closure for window {h["id"]}')


def stats(values):
    v = sorted(values)
    return {'total_ms': sum(v) / 1000000, 'p50_ms': v[(len(v)-1)//2] / 1000000,
            'p95_ms': v[math.ceil(.95*len(v))-1] / 1000000,
            'p99_ms': v[math.ceil(.99*len(v))-1] / 1000000, 'max_ms': v[-1] / 1000000}


def analyze(path, slow_ms=100):
    rounds, hosts = {}, {}
    with Path(path).open(errors='replace') as f:
        for line in f:
            if line.startswith('batch-round '):
                # Slot fragments repeat rows=; never overwrite the window rows.
                r = {k: float(v) for k, v in KV.findall(line.split(' slot=', 1)[0])}
                ident = int(r['id'])
                if ident in rounds:
                    raise ValueError(f'duplicate batch-round id {ident}')
                rounds[ident] = r
            elif line.startswith('batch-host '):
                h = {k: int(v) for k, v in KV.findall(line)}  # preserve raw 64-bit ns
                if h['id'] in hosts:
                    raise ValueError(f'duplicate batch-host id {h["id"]}')
                hosts[h['id']] = h
    if not hosts or rounds.keys() != hosts.keys():
        raise ValueError('host/round coverage incomplete or profiling disabled')
    for ident, h in hosts.items():
        validate(h, rounds[ident])
    if any(b['end_mono_ns'] < a['end_mono_ns'] for a, b in
           zip(list(hosts.values()), list(hosts.values())[1:])):
        raise ValueError('nonmonotonic host timestamp')
    outer = ('admin_ns', 'return_ns', 'run_other_ns') + RUN
    children = ('stage_other_ns', 'ple_gather_ns') + STAGE
    tops = []
    for ident, h in sorted(hosts.items(), key=lambda item: item[1]['verify_ns'], reverse=True)[:10]:
        r = rounds[ident]
        dominant = max(outer, key=lambda key: h[key])
        tops.append({'id': ident, 'rows': h['rows'], 'active': int(r['active']),
                     'head_type': int(r['head_type']), 'head_wide_active': int(r['head_wide_active']),
                     'verify_ms': h['verify_ns']/1000000, 'dominant_host_bin': dominant,
                     'dominant_stage_child': max(('stage_other_ns',) + STAGE, key=lambda key: h[key]),
                     'host_bins_ms': {k: h[k]/1000000 for k in outer + children},
                     'graph_cache_hit': h['graph_cache_hit'], 'end_mono_ns': h['end_mono_ns']})
    return {'log': str(path), 'windows': len(hosts), 'closure_pass': True,
            'slow_windows': sum(h['verify_ns'] >= slow_ms*1000000 for h in hosts.values()),
            'host_phase_stats': {k: stats([h[k] for h in hosts.values()]) for k in outer + children},
            'top_windows': tops}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', nargs='+', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--slow-ms', type=float, default=100)
    args = p.parse_args()
    result = {'schema': 'strata.batch-host.v1', 'unit': 'host wall ms, raw counters ns',
              'policy': 'diagnostic only; tail_sync may be GPU/driver/wakeup wait, not a kernel attribution',
              'logs': [analyze(path, args.slow_ms) for path in args.logs]}
    text = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end='')
