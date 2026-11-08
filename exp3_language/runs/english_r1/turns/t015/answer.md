<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def orth(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:r]
q,k,v,o=hw(0,2); A=q.T@k; u,s,vt=np.linalg.svd(A)
P=orth(np.hstack([u[:,:18],vt[:18].T]),20)
ov2=o@v; uo,so,vo=np.linalg.svd(ov2)
T=vo[:20].T; O2=uo[:,:20]
Ob=orth(np.hstack([np.linalg.svd(hw(0,i)[3]@hw(0,i)[2])[0][:,:3] for i in [0,1,4,5]]),8)
subs=dict(P=P,T=T,O2=O2,Ob=Ob)
for L,j in [(3,0),(3,1),(3,5),(2,0),(1,2)]:
    q,k,v,o=hw(L,j); QK=q.T@k
    base=np.linalg.norm(QK)/384
    print('L%dH%d'%(L,j))
    for a,Sa in subs.items():
        row=[]
        for b,Sb in subs.items():
            B=Sa.T@QK@Sb
            tr=np.trace(B)/np.sqrt(B.shape[0]*B.shape[1]) if a in('T','O2') and b in ('T','O2') else 0
            row.append('%s:%.1f'%(b,np.linalg.norm(B)/np.sqrt(B.size)/base))
        print('  q',a,' '.join(row))
    # trace on T x T, and on T(query) x O2(key) mapped back via OV2
    M=T.T@QK@ov2@T  # query token -> key's prev token (in T coords)
    ev=np.linalg.eigvals(M); print('  T x O2(OV2) eig-score %.2f'%(ev.real.sum()/np.abs(ev).sum()), ' TxT eig-score %.2f'%(np.linalg.eigvals(T.T@QK@T).real.sum()/np.abs(np.linalg.eigvals(T.T@QK@T)).sum()))
```

````