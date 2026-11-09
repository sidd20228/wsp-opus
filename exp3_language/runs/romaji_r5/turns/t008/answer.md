```python
import numpy as np
np.set_printoptions(linewidth=220)
d=384; hd=64
def heads(l):
    w=W[f'ln1_w.{l}']; res=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        res.append((W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]))
    return res
H={l:heads(l) for l in range(4)}
A={(l,j):H[l][j][0].T@H[l][j][1] for l in range(4) for j in range(6)}
OV={(l,j):H[l][j][3]@H[l][j][2] for l in range(4) for j in range(6)}
def evs(M,n=64):
    ev=np.linalg.eigvals(M); return np.array(sorted(ev,key=lambda z:-abs(z)))[:n]
u,s,vt=np.linalg.svd(OV[(0,3)]); T3=vt[:20]       # token-ish input subspace read by L0H3
u0,s0,vt0=np.linalg.svd(OV[(0,0)]); T0=vt0[:12]
T=np.linalg.qr(np.vstack([T3,T0]).T)[0].T          # 32-dim
print('T dims',T.shape)
def ratio(G):
    e=evs(G,len(G)); return e.real.sum()/np.abs(e).sum(), np.abs(e).sum()
for L in range(1,4):
    for h in range(6):
        Ah=A[(L,h)]
        Gsame=T@Ah@T.T
        Gind3=T@Ah@OV[(0,3)]@T.T
        Gind0=T@Ah@OV[(0,0)]@T.T
        r1,m1=ratio(Gsame); r2,m2=ratio(Gind3); r3,m3=ratio(Gind0)
        print(f'L{L}H{h} same-token: ratio {r1:+.2f} mag {m1:6.2f} | via L0H3: ratio {r2:+.2f} mag {m2:6.2f} | via L0H0: ratio {r3:+.2f} mag {m3:6.2f}')
# relation of T to the 2D dominant subspace and to positional q3
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
for nm,X in [('v5',topRows(H[0][5][2],3)),('q3',topRows(H[0][3][0],20)),('k3',topRows(H[0][3][1],20)),('fc',topRows(W['W_fc.0']*W['ln2_w.0'],40))]:
    print(nm,'overlap with T, principal cos:',np.round(np.linalg.svd(X@T.T,compute_uv=False)[:10],2))
```