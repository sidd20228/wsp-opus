```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
sl=lambda j: slice(j*hd,(j+1)*hd)
def rd(nm,l,j): return W[f'{nm}.{l}'][sl(j)]*W[f'ln1_w.{l}']
fc=W['W_fc.0']*W['ln2_w.0']
U=np.vstack([topRows(rd('W_q',0,3),34),topRows(rd('W_k',0,3),34)]); u,s,vt=np.linalg.svd(U)
for npos in [0,20,26,30,34]:
    for nout in [(0,0),(10,2),(20,3),(25,4),(32,6)]:
        outs=[vt[:npos]] if npos>0 else []
        for j in range(6):
            OVj=W['W_o.0'][:,sl(j)]@rd('W_v',0,j); uu,ss,vv=np.linalg.svd(OVj)
            r=nout[0] if j in (0,3) else nout[1]
            if r>0: outs.append(uu[:,:r].T)
        if outs:
            Bad=np.linalg.qr(np.vstack(outs).T)[0].T; Proj=np.eye(384)-Bad.T@Bad
        else: Proj=np.eye(384)
        sv=np.linalg.svd(fc@Proj,compute_uv=False)
        ratios=sv[:40]/sv[1:41]
        k=np.argmax(ratios[15:35])+15
        print(npos,nout,'removed',384-int(round(np.trace(Proj))),'sv[20:32]',np.round(sv[20:32],2),' biggest drop after index',k,'(count',k+1,') ratio %.2f'%ratios[k])
```