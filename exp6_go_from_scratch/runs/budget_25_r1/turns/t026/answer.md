The game-62 and game-59 blunders are now avoided; game 123's parity-collision fake still has 13%. I'll inspect the feature activations at a few points to confirm the units compute what I intended.

```python
import numpy as np
exec(open('/work/build5.py').read())
d='/work/pass_4/'
L=np.load(d+'logits.npy'); A6=np.load(d+'act_L6.npy'); A5=np.load(d+'act_L5.npy')
def feats(i,e):
    a=A6[i]
    return dict(LIN=a[LIN+e]-10,SA=a[SA+e],SA2=a[SA2+e],NE=a[NE+e],INFP=a[INFP+e],cap=sum(a[N5+n] for n in NB[e]),logit=L[i,e])
for i,e in [(7,36),(7,74),(6,53),(6,6),(8,0),(4,17),(4,80)]:
    print(i,divmod(e,9),{k:round(float(v),2) for k,v in feats(i,e).items()})
i=7
print('AO',[divmod(p,9) for p in range(81) if A5[i,AO+p]>0])
print('S3 count',int((A5[i,S3:S3+81]>0).sum()),'own count',int((A5[i,OWN:OWN+81]>0).sum()))
print('NX atari (4hop) opp stones',[divmod(p,9) for p in range(81) if A5[i,(NX0 if PAR[p]==0 else NX1)+p]>0 and A5[i,OPP+p]>0][:20])
print('N5 atari',[divmod(p,9) for p in range(81) if A6[i,N5+p]>0][:20])
```