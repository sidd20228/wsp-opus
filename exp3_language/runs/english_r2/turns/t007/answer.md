```python
def head(l,j):
    s=slice(64*j,64*j+64)
    w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
QK={};OV={}
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=head(l,j); QK[l,j]=Wq.T@Wk; OV[l,j]=Wo@Wv
U={}
for j in range(6):
    u,s,vt=np.linalg.svd(QK[0,j]); U['qL',j]=u[:,0]; U['qR',j]=vt[0]
    u,s,vt=np.linalg.svd(OV[0,j]); U['oL',j]=u[:,0]; U['oR',j]=vt[0]
keys=[(t,j) for j in [0,1,4,5,3] for t in ['qL','qR','oR','oL']]
M=np.array([[abs(U[a]@U[b]) for b in keys] for a in keys])
print(keys)
print(M)
```