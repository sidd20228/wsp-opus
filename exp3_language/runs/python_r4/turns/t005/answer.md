```python
import numpy as np
hd=64
def heads(l):
    lw=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]*lw; Wk=W[f'W_k.{l}'][s]*lw; Wv=W[f'W_v.{l}'][s]*lw; Wo=W[f'W_o.{l}'][:,s]
        out.append(dict(QK=Wq.T@Wk, OV=Wo@Wv))
    return out
H=[heads(l) for l in range(4)]
rng=np.random.default_rng(0)
def score(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
# baseline: random orthogonal rotation of B
def base(A,B,n=5):
    vals=[]
    for _ in range(n):
        Q,_=np.linalg.qr(rng.standard_normal((384,384)))
        vals.append(score(A,Q@B@Q.T))
    return np.mean(vals)
for l2 in range(1,4):
    for l1 in range(l2):
        print(f'--- reader layer {l2} <- writer layer {l1}  (rows: reader head, cols writer head); K, Q, V comp ratio to baseline')
        for j2 in range(6):
            K=[];Qc=[];V=[]
            for j1 in range(6):
                OV=H[l1][j1]['OV']; QK=H[l2][j2]['QK']; OV2=H[l2][j2]['OV']
                b=base(QK,OV,2)
                K.append(score(QK,OV)/b)
                Qc.append(score(QK.T,OV)/b)
                V.append(score(OV2,OV)/base(OV2,OV,2))
            print(j2,'K',np.round(K,1),'Q',np.round(Qc,1),'V',np.round(V,1))
```