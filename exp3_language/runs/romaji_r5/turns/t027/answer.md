```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
sl=lambda j: slice(j*hd,(j+1)*hd)
def rd(nm,l,j): return W[f'{nm}.{l}'][sl(j)]*W[f'ln1_w.{l}']
U=np.vstack([topRows(rd('W_q',0,3),30),topRows(rd('W_k',0,3),30)]); u,s,vt=np.linalg.svd(U); P=vt[:30]
G=topRows(rd('W_v',0,5),2)   # dominant positional plane g
print('g in P:',np.round(np.linalg.norm(G@P.T,axis=1),2))
def plane_of(M):
    ev,V=np.linalg.eig(M); i=np.argmax(np.abs(ev)*(np.abs(ev.imag)>1e-6))
    X=np.stack([V[:,i].real,V[:,i].imag]); Q=np.linalg.qr(X.T)[0].T; return ev[i],Q
for L in range(4):
    for h in range(6):
        Ah=rd('W_q',L,h).T@rd('W_k',L,h)
        B=P@Ah@P.T
        ev,Q=plane_of(B)
        Q384=Q@P
        cg=np.linalg.svd(Q384@G.T,compute_uv=False)
        # also 2x2 restricted to g plane
        Mg=G@Ah@G.T
        ang=np.degrees(np.arctan2(Mg[1,0]-Mg[0,1],Mg[0,0]+Mg[1,1])); mag=np.hypot(Mg[1,0]-Mg[0,1],Mg[0,0]+Mg[1,1])/2
        print(f'L{L}H{h} top complex PP eig {abs(ev):.2f}@{np.degrees(np.angle(ev)):.0f}  plane-vs-g cos {np.round(cg,2)}   g-plane rot {ang:.0f}deg mag {mag:.2f} sym {np.round(np.linalg.eigvalsh((Mg+Mg.T)/2),2)}')
```