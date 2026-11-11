The structure is clear: each sample has correct, cut, pred and tokens. Next I compute paired accuracy differences of gen_12 (brake + gate) against gen_9 (brake + E1) and gen_8 (brake only), overall, by source and by split (setA/setB from splits.json). I also tabulate the outcome transitions (finished-correct, finished-wrong, no-answer, cut). A paired gain over gen_9 of more than 3 SE, consistent in both splits, would make this the new best edit.

```python
# Paired comparison gen_12 (x5+gate) vs gen_9 (x5+E1) and gen_8 (x5), by source and split; outcome transitions.
sp = json.load(open('/work/splits.json')); print({k:len(v) for k,v in sp.items()})
def table(g):
    d={}
    for r in g:
        for i,s in enumerate(r['samples']):
            st = 'cut' if s['cut'] else ('ok' if s['correct'] else ('noans' if s['pred'] in (None,'') else 'wrong'))
            d[(r['id'],i)] = dict(c=int(s['correct']), st=st, src=r['source'], t=s['tokens'])
    return d
T12,T9,T8 = table(g12),table(g9),table(g8)
setA = set(sp.get('setA',[])); setB=set(sp.get('setB',[]))
def paired(Ta,Tb,filt=lambda k,v:True):
    ks=[k for k in Ta if k in Tb and filt(k,Ta[k])]
    d=np.array([Ta[k]['c']-Tb[k]['c'] for k in ks])
    return f"n={len(ks)} diff={100*d.mean():+.1f}+-{100*d.std(ddof=1)/np.sqrt(len(ks)):.1f} w->r={int((d>0).sum())} r->w={int((d<0).sum())}"
for nm,Tb in [('vs gen_9 x5+E1',T9),('vs gen_8 x5',T8)]:
    print(nm, 'all', paired(T12,Tb))
    for src in ['math','gsm8k']: print('   ',src, paired(T12,Tb,lambda k,v:v['src']==src))
    print('    setA', paired(T12,Tb,lambda k,v:k[0] in setA), '| setB', paired(T12,Tb,lambda k,v:k[0] in setB))
tr = collections.Counter((T9[k]['st'],T12[k]['st']) for k in T12)
print('transitions gen9->gen12:', sorted(tr.items(), key=lambda x:-x[1]))
print('status counts gen12:', collections.Counter(v['st'] for v in T12.values()))
print('mean tokens finished gen12/gen9:', np.median([v['t'] for v in T12.values() if v['st']!='cut']), np.median([v['t'] for v in T9.values() if v['st']!='cut']))
```