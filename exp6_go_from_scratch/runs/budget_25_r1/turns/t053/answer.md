Dame are only 0–6 points and almost no opponent stones are left in atari, so the score losses come from the territory split itself. I worked out new self-atari constants that separate the cases that matter:
- An E=1 capture stays unpenalised.
- A fake-capture E=0 connection to a weak chain gets penalised (3.5 units).
- Single-stone self-atari, futile atari extensions and rescues keep their earlier values.

First, let me check how the B openings go.

```python
import json, numpy as np
d='/work/pass_9/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
for gi,g in enumerate(G[:12]):
    nm=[(divmod(m['move'],9), m['top5'][0][1]) for m in g['moves'] if m['by']=='net'][:5]
    om=[divmod(m['move'],9) for m in g['moves'] if m['by']=='opponent'][:5]
    print(gi,g['opponent'][:2],g['net_color'][0],g['result'],'net',nm,'opp',om)
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
print('\n'.join(bs(B[0,19])))
```