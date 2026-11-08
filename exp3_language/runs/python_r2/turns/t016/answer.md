```python
import numpy as np
hd=64
def colsp(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k]
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
D=[W[f'ln1_w.{l}'] for l in range(4)]
sl=lambda h: slice(h*hd,(h+1)*hd)
OV0=lambda h: W['W_o.0'][:,sl(h)]@(W['W_v.0'][sl(h)]*D[0])
subs={}
subs['pos']=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D[0],12)
subs['tok']=rowsp(np.vstack([W['W_v.0'][sl(1)],W['W_v.0'][sl(5)]])*D[0],20)
subs['avg04']=colsp(np.hstack([OV0(0),OV0(4)]),3)
subs['prev1']=colsp(OV0(1),20)
subs['prev5']=colsp(OV0(5),20)
subs['h2']=colsp(OV0(2),6); subs['h3']=colsp(OV0(3),6)
subs['mlp0']=colsp(W['W_proj.0'],60)
names=list(subs)
def strength(Wm,B):  # mean sq norm per dim of Wm@B relative to average per dim
    return (np.linalg.norm(Wm@B)**2/B.shape[1])/(np.linalg.norm(Wm)**2/384)
print('head side '+' '.join('%6s'%n for n in names))
for l in [1,2,3]:
    for h in range(6):
        for side in ['W_q','W_k','W_v']:
            Wm=W[f'{side}.{l}'][sl(h)]*D[l]
            print(f'L{l}H{h} {side[-1]} '+' '.join('%6.2f'%strength(Wm,subs[n]) for n in names))
```