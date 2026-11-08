```python
def head(l,j):
    s=slice(64*j,64*j+64)
    w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
QK={};OV={}
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=head(l,j); QK[l,j]=Wq.T@Wk; OV[l,j]=Wo@Wv
u,s,vt=np.linalg.svd(OV[0,2]); k=24
Bo=u[:,:k]   # output space of prev head
Bi=vt[:k].T  # input (token) space of prev head
def fr(M,B,side):
    return (np.linalg.norm(B.T@M)**2 if side=='q' else np.linalg.norm(M@B)**2)/np.linalg.norm(M)**2
print('baseline',k/384)
for l in range(4):
    for j in range(6):
        M=QK[l,j]; Sym=(M+M.T)/2
        print(f'L{l}H{j} keyside in OV02out {fr(M,Bo,"k"):.2f} q-side in OV02out {fr(M,Bo,"q"):.2f} | q-side in tok {fr(M,Bi,"q"):.2f} k-side in tok {fr(M,Bi,"k"):.2f} | trace/fro {np.trace(M)/np.linalg.norm(M):+.2f} symfrac {np.linalg.norm(Sym)**2/np.linalg.norm(M)**2:.2f}')
```