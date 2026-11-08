```python
import numpy as np
H=6;hd=64
D=[W[f'ln1_w.{l}'] for l in range(4)]
def sub(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
s1=slice(64,128); s5=slice(320,384)
Vcomb=np.vstack([W['W_v.0'][s1]*D[0], W['W_v.0'][s5]*D[0]])
T=sub(Vcomb,20)
def QK(l,h):
    s=slice(h*hd,(h+1)*hd); return (W[f'W_q.{l}'][s]*D[l]).T @ (W[f'W_k.{l}'][s]*D[l])
def OV(l,h):
    s=slice(h*hd,(h+1)*hd); return W[f'W_o.{l}'][:,s] @ (W[f'W_v.{l}'][s]*D[l])
def pos(M):
    e=np.linalg.eigvals(M); return e.real.sum()/np.abs(e).sum()
for l2 in [1,2,3]:
    for b in range(H):
        row=[]
        for a in range(H):
            M=T.T@QK(l2,b)@OV(0,a)@T
            row.append('%+.2f(%.1f)'%(pos(M),np.linalg.norm(M)))
        # direct token-token
        Md=T.T@QK(l2,b)@T
        print('L%dH%d'%(l2,b),' '.join(row),' direct %+.2f(%.1f)'%(pos(Md),np.linalg.norm(Md)))
```