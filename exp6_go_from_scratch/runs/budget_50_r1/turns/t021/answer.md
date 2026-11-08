Let me verify v2 with a positions pass: earlier tests plus positions from the last games where my net played a self-atari.

```python
exec(open('/work/ana.py').read())
from collections import Counter
G=json.load(open('/work/pass_2/games.json')); Bd=np.load('/work/pass_2/boards.npy')
d=json.load(open('/work/tests1.json'))
def tob(arr): return [1 if x==1 else (2 if x==-1 else 0) for x in arr]
sa_pos=[]
for gi,g in enumerate(G):
    netc=1 if g['net_color']=='black' else 2
    prev=[0]*81
    for k,m in enumerate(g['moves']):
        b=tob(Bd[gi,k])
        if m['by']=='net' and m['move']<81:
            p=m['move']; capt=sum(1 for q in range(81) if prev[q]==3-netc and b[q]==0)
            gg,l=group(b,p)
            if len(l)==1 and capt==0 and len(gg)>=2:
                sa_pos.append((gi,k,p))
        prev=b
import random; random.seed(3); random.shuffle(sa_pos); sa_pos=sa_pos[:24]
posl=[{"board":b,"to_move":tm} for (_,b,tm) in d['T']]+[{"moves":[("pass" if x==81 else x) for x in m]} for (_,m) in d['R']]
meta=[]
for gi,k,p in sa_pos:
    mv=[x['move'] for x in G[gi]['moves'][:k]]
    posl.append({"moves":[("pass" if x==81 else x) for x in mv]}); meta.append((gi,k,p,mv))
json.dump({'T':d['T'],'R':d['R'],'SA':meta},open('/work/tests2.json','w'))
print(len(posl))
print(json.dumps({"positions":posl,"activations":True},separators=(',',':')))
```