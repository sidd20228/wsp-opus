```python
import numpy as np
hd=64
D=[W[f'ln1_w.{l}'] for l in range(4)]
B=[W[f'ln1_b.{l}'] for l in range(4)]
sl=lambda h: slice(h*hd,(h+1)*hd)
def rsvd(M):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt,S
V1,_=rsvd(W['W_v.0'][sl(1)])
Tx,_=np.linalg.qr(D[0][:,None]*V1[:27].T)
G={}
for l in range(4):
    for h in range(6):
        qb=W[f'W_q.{l}'][sl(h)]@B[l]
        Kt=(W[f'W_k.{l}'][sl(h)]*D[l])@Tx   # 64x27
        g=Kt.T@qb
        sv=np.linalg.svd(Kt,compute_uv=False)
        G[(l,h)]=g
        print(f'L{l}H{h} |qb|=%.2f static key-token pref norm=%.3f  Ktok sv'%(np.linalg.norm(qb),np.linalg.norm(g)),np.round(sv[:5],2))
# compare static preference directions across heads
keys=[k for k in G if np.linalg.norm(G[k])>0.1]
import itertools
for a,b in itertools.combinations(keys,2):
    c=G[a]@G[b]/np.linalg.norm(G[a])/np.linalg.norm(G[b])
    if abs(c)>0.6: print(a,b,'cos %.2f'%c)
```