One loss vs weak. Let me check if it's the repetition one, and test α=1.2 in the same run.

```python
import json, numpy as np
G=json.load(open('/work/pass_41/games.json')); B=np.load('/work/pass_41/boards.npy')
for gi,g in enumerate(G):
    if g['result']==-1 or g['end']=='repetition':
        n=len(g['moves']); s=f"{gi} {g['opponent']} {g['net_color']} {g['end']} res {g['result']} margin {g['margin']}"
        if g['end']=='repetition':
            last=B[gi,n-1]; k0=[k for k in range(n-1) if np.array_equal(B[gi,k],last)][-1]
            s+=' cyc'+str([(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', round(m['top5'][0][1],2) if 'top5' in m else '') for m in g['moves'][k0+1:n]])
        print(s)
exec(open('/work/build8.py').read())
cfg=json.load(open('/work/cfg_v33.json')); cfg.update(alpha=1.2)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v34.json','w'))
print('saved v34 alpha=1.2')
```