```python
import numpy as np
hd=64
lw=W['ln1_w.0']
def QKh(j):
    s=slice(j*hd,(j+1)*hd)
    return (W['W_q.0'][s]*lw).T@(W['W_k.0'][s]*lw)
def planes(M,n):
    ev,vec=np.linalg.eig(M)
    idx=np.argsort(-np.abs(ev))
    res=[];seen=[]
    for i in idx:
        e=ev[i]
        if e.imag< -1e-9: continue
        v=vec[:,i]
        if abs(e.imag)>1e-9:
            B=np.stack([v.real,v.imag],1)
        else:
            B=v.real[:,None]
        Q,_=np.linalg.qr(B)
        res.append((abs(e),np.degrees(np.angle(e)),Q))
        if len(res)>=n: break
    return res
P1=planes(QKh(1),8); P5=planes(QKh(5),14)
for a,th,Q in P1:
    best=[]
    for b,th2,Q2 in P5:
        c=np.linalg.svd(Q.T@Q2,compute_uv=False)
        best.append((c.mean(),b,th2))
    best.sort(reverse=True)
    print('H1 %.2f@%d dim%d'%(a,th,Q.shape[1]),' best H5 matches:',[('%.2f'%c,'%.2f@%d'%(b,t)) for c,b,t in best[:3]])
# Also overlap with L0H4 read plane
OV4=W['W_o.0'][:,4*hd:5*hd]@(W['W_v.0'][4*hd:5*hd]*lw)
_,_,Vt=np.linalg.svd(OV4); R=Vt[:2].T
for a,th,Q in P1+P5:
    print('%.2f@%d'%(a,th),'overlap with H4 read plane %.2f'%np.linalg.norm(R.T@Q)**2)
```