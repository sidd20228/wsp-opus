```python
svd=np.linalg.svd
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
# Note: layer-3 head weights are in z3 coords (LN3 normalized, scaled by ln1_w.3); layer0 in z0 coords (scaled by ln1_w.0).
# Compare in raw residual coords: unscaled directions. Use raw W_v rows (no g) for subspaces.
def rowspace(M,k):
    U,s,Vt=svd(M); return Vt[:k].T, s
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
# raw-coord subspaces: OV_raw = Wo Wv diag(g) -> right sing vectors in raw coords already includes g weighting; fine.
E31,s31=rowspace(OV[(3,1)],30); print('3.1 OV sv',s31[:40].round(2))
E35,s35=rowspace(OV[(3,5)],30); print('3.5 OV sv',s35[:40].round(2))
U,s,Vt=svd(H[(0,2)][0].T@H[(0,2)][1]); S=np.linalg.qr(np.hstack([Vt[:18].T,U[:,:18]]))[0][:,:20]
T,_=rowspace(OV[(0,2)],20)
_,_,Vtu=svd(OV[(0,0)]+OV[(0,1)]+OV[(0,4)]+OV[(0,5)]); Vu=Vtu[:3].T
def ov(A,B): return np.linalg.svd(A.T@B,compute_uv=False).round(2)
for name,E in [('E31',E31),('E35',E35)]:
    for k in [10,20,28]:
        print(name,k,'vs S',ov(E[:,:k],S)[:8],'vs T',ov(E[:,:k],T)[:8],'vs Vu',ov(E[:,:k],Vu))
```