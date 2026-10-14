#!/usr/bin/env python3
"""Validate actual captured QSA dispatch counts on the frozen 12-QSA-layer model."""
import argparse
import json
from pathlib import Path
from analyze_batch_host_profile import KV


def validate(r):
    required=('rows','qsa_tile_active','qsa_tile_calls','qsa_tile_rows','qsa_scalar_rows')
    if set(required)-r.keys() or any(type(r[k]) is not int or r[k]<0 for k in required):
        raise ValueError('missing/noninteger/negative QSA counter')
    c,t,s=r['qsa_tile_calls'],r['qsa_tile_rows'],r['qsa_scalar_rows']
    if not (r['rows']>0 and r['qsa_tile_active']==int(c>0) and 2*c<=t<=8*c and
            t+s==12*r['rows'] and c%12==0 and t%12==0 and s%12==0):
        raise ValueError('QSA actual-dispatch closure failed')


def analyze(path):
    rows={}
    with Path(path).open(errors='replace') as f:
        for line in f:
            if not line.startswith('batch-round '):continue
            r={k:(float(v) if '.' in v else int(v)) for k,v in KV.findall(line.split(' slot=',1)[0])}
            validate(r)
            if r['id'] in rows:raise ValueError('duplicate window')
            rows[r['id']]=r
    if not rows:raise ValueError('no native windows')
    tiles=sum(r['qsa_tile_calls'] for r in rows.values()); packed=sum(r['qsa_tile_rows'] for r in rows.values())
    scalar=sum(r['qsa_scalar_rows'] for r in rows.values())
    return {'windows':len(rows),'dispatch_closure_pass':True,'actual_tile_calls':tiles,'actual_tile_rows':packed,
            'scalar_rows':scalar,'tile_row_coverage':packed/(packed+scalar),'eligible_active_windows':sum(r['qsa_tile_active'] for r in rows.values()),
            'removed_selection_attention_kernel_nodes':4*(packed-tiles),
            'scope':'actual captured full-Q4 tile dispatch; kernel nodes, not measured GPU time/DRAM bytes'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();text=json.dumps(analyze(a.log),indent=2)+'\n'
    if a.out:a.out.write_text(text)
    else:print(text,end='')
