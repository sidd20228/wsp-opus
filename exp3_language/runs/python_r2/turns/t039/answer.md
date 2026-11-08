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
QK=lambda l,h:(W[f'W_q.{l}'][sl(h)]*D[l]).T@(W[f'W_k.{l}'][sl(h)]*D[l])
G={}
for l in range(4):
    for h in range(6):
        qb=W[f'W_q.{l}'][sl(h)]@B[l]
        G[(l,h)]=((W[f'W_k.{l}'][sl(h)]*D[l])@Tx).T@qb
gL1=np.mean([G[(1,j)]/np.linalg.norm(G[(1,j)]) for j in range(6)],0); gL1/=np.linalg.norm(gL1)
for l in [1,2,3]:
    for h in range(6):
        M=Tx.T@QK(l,h)@Tx
        rq=gL1@M@gL1; avg=np.trace(M)/27
        # also query-side: how query tau changes preference: M^T? q=tau: scores over keys = tau^T M
        print(f'L{l}H{h} R(g)=%.3f avg diag=%.3f ratio=%.1f   |static g|=%.2f cos(g_lh,gL1)=%.2f'%(rq,avg,rq/avg if avg!=0 else 0,np.linalg.norm(G[(l,h)]),G[(l,h)]@gL1/np.linalg.norm(G[(l,h)])))
```