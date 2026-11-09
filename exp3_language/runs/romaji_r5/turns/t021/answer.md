```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
sl=lambda j: slice(j*hd,(j+1)*hd)
def rd(nm,l,j): return W[f'{nm}.{l}'][sl(j)]*W[f'ln1_w.{l}']
U=np.vstack([topRows(rd('W_q',0,3),30),topRows(rd('W_k',0,3),30)]); u,s,vt=np.linalg.svd(U); P=vt[:26]
fc=W['W_fc.0']*W['ln2_w.0']
Proj=np.eye(384)-P.T@P
A1=topRows(rd('W_v',0,3)@Proj,25); B1=topRows(fc@Proj,50)
uu,ss,vv=np.linalg.svd(A1@B1.T); Tc=(uu.T@A1)[:24]
def evs(M,n=64):
    ev=np.linalg.eigvals(M); return np.array(sorted(ev,key=lambda z:-abs(z)))[:n]
for L in range(1,4):
    for h in range(6):
        Ah=rd('W_q',L,h).T@rd('W_k',L,h)
        G=Tc@Ah@Tc.T
        e=evs(G,24)
        print(f'L{L}H{h} TT eig:',' '.join('%.2f@%.0f'%(abs(z),np.degrees(np.angle(z))) for z in e[:16] if np.angle(z)>=0))
# same for OV (token->token) of L1 heads: Tc OV Tc^T
for L in range(1,4):
    for h in range(6):
        OVh=W[f'W_o.{L}'][:,sl(h)]@rd('W_v',L,h)
        e=evs(Tc@OVh@Tc.T,24)
        print(f'L{L}H{h} OV TT eig:',' '.join('%.2f@%.0f'%(abs(z),np.degrees(np.angle(z))) for z in e[:10] if np.angle(z)>=0), ' frac OV in T->T %.2f'%(np.linalg.norm(Tc@OVh@Tc.T)/np.linalg.norm(OVh)))
```