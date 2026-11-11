```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Q3,K3,V3,O3=head(0,3);OV3=O3@V3;M3=Q3.T@K3
ev,evec=np.linalg.eig(M3);idx=np.argsort(-np.abs(ev))[:20]
Bp,_=np.linalg.qr(np.hstack([evec[:,idx].real,evec[:,idx].imag]))
uo,so,vo=np.linalg.svd(OV3)
print('L0H3 OV sv',np.round(so[:32],2))
T=vo[:26].T
print('L0H3 OV input (top26) frac in its own QK eig-planes: %.2f'%(fro(Bp.T@T)**2/26), ' random exp %.2f'%(Bp.shape[1]/384))
print('L0H3 QK planes frac in its OV input space: %.2f'%(fro(T.T@Bp)**2/Bp.shape[1]))
# OV3 in-place copy? eigen of T^T OV3 T
e=np.linalg.eigvals(T.T@OV3@T);print('OV3 in-place eig realfrac %.2f'%(e.real.sum()/np.abs(e).sum()), 'out-space overlap with in-space %.2f'%(fro(uo[:,:26].T@T)**2/26))
def idc(M): return np.trace(M)/fro(M)/np.sqrt(M.shape[0])
for l in range(1,4):
    row=[]
    for j in range(H):
        Q,K,V,O=head(l,j);QK=Q.T@K
        Mi=T.T@QK@OV3@T; Md=T.T@QK@T
        row.append('H%d ind %+.2f (|%.2f|) same %+.2f (|%.2f|)'%(j,idc(Mi),fro(Mi),idc(Md),fro(Md)))
    print('L%d '%l+' | '.join(row))
Q,K,V,O=head(3,3);Mi=T.T@Q.T@K@OV3@T
e=np.linalg.eigvals(Mi);e=e[np.argsort(-np.abs(e))]
print('L3H3 ind eig',' '.join('%.2f@%d'%(abs(x),np.degrees(np.angle(x))) for x in e[:14]))
```