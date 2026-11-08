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
        G[(l,h)]=((W[f'W_k.{l}'][sl(h)]*D[l])@Tx).T@qb
# L0 heads: key token direction (top right sv of K restricted to T)
for h in [0,2,3,1,5]:
    Kt=(W['W_k.0'][sl(h)]*D[0])@Tx
    U,S,Vt=np.linalg.svd(Kt)
    gL1=np.mean([G[(1,j)]/np.linalg.norm(G[(1,j)]) for j in range(6)],0); gL1/=np.linalg.norm(gL1)
    print('L0H%d key-token top dirs vs L1 static pref: '%h, np.round(Vt[:3]@gL1,2), ' static g0 cos %.2f'%(G[(0,h)]@gL1/np.linalg.norm(G[(0,h)])))
# symmetric part of L1 token-token form: most negative eigvec vs gL1
QK=lambda l,h:(W[f'W_q.{l}'][sl(h)]*D[l]).T@(W[f'W_k.{l}'][sl(h)]*D[l])
for h in range(6):
    M=Tx.T@QK(1,h)@Tx; Ms=(M+M.T)/2
    e,V=np.linalg.eigh(Ms)
    print('L1H%d most-neg eig %.2f, cos with gL1: %.2f ; 2nd %.2f cos %.2f; most-pos %.2f cos %.2f'%(h,e[0],V[:,0]@gL1,e[1],V[:,1]@gL1,e[-1],V[:,-1]@gL1))
```