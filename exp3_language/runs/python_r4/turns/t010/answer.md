```python
import numpy as np
hd=64
def colspace(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return U[:,:k],S[:k]
# writer subspaces
lw0=W['ln1_w.0']
OV0=[W['W_o.0'][:,j*hd:(j+1)*hd]@(W['W_v.0'][j*hd:(j+1)*hd]*lw0) for j in range(6)]
lw1=W['ln1_w.1']; lb1=W['ln1_b.1']
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    Wq=W['W_q.1'][s]; Wk=W['W_k.1'][s]
    qb=Wq@lb1
    kp=(Wk*lw1).T@qb   # key preference in x-hat space (normalized residual)
    out=[]
    for h in range(6):
        U,S=colspace(OV0[h],5)
        out.append(np.linalg.norm(U.T@kp)/np.linalg.norm(kp))
    Up,Sp=colspace(W['W_proj.0'],30)
    print(j,'|kp| %.2f'%np.linalg.norm(kp),'frac in top5 of L0 head OV out:',np.round(out,2),'in MLP0 top30 %.2f'%(np.linalg.norm(Up.T@kp)/np.linalg.norm(kp)))
# K-comp of L1 with MLP0 output
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    Wk=W['W_k.1'][s]*lw1; Wq=W['W_q.1'][s]*lw1
    Wp=W['W_proj.0']
    def ratio(A,B):
        return (np.linalg.norm(A@B)**2/np.linalg.norm(B)**2)/(np.linalg.norm(A)**2/384)
    print(j,'K-read of MLP0 out %.2f, Q-read %.2f; K-read of L0H4 %.2f L0H0 %.2f L0H1 %.2f L0H5 %.2f'%(ratio(Wk,Wp),ratio(Wq,Wp),ratio(Wk,OV0[4]),ratio(Wk,OV0[0]),ratio(Wk,OV0[1]),ratio(Wk,OV0[5])))
```