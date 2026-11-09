```python
import numpy as np
hd=64
lw=W['ln1_w.0']
def QKh(j):
    s=slice(j*hd,(j+1)*hd)
    return (W['W_q.0'][s]*lw).T@(W['W_k.0'][s]*lw)
def OV0(j): return W['W_o.0'][:,j*hd:(j+1)*hd]@(W['W_v.0'][j*hd:(j+1)*hd]*lw)
def eigdirs(M,n):
    ev,vec=np.linalg.eig(M); idx=np.argsort(-np.abs(ev)); res=[]
    for i in idx:
        if ev[i].imag<-1e-9: continue
        v=vec[:,i]; Bm=np.stack([v.real,v.imag],1) if abs(ev[i].imag)>1e-9 else v.real[:,None]
        Q,_=np.linalg.qr(Bm); res.append((ev[i],Q))
        if len(res)>=n: break
    return res
D1=eigdirs(QKh(1),4); D5=eigdirs(QKh(5),6)
_,_,V4=np.linalg.svd(OV0(4)); P4=V4[:2].T
for j in [0,2,3]:
    for e,Q in eigdirs(QKh(j),3):
        ov=[ '%.2f'%(np.linalg.norm(Q.T@Q2)**2/Q.shape[1]) for e2,Q2 in D1]
        ov5=[ '%.2f'%(np.linalg.norm(Q.T@Q2)**2/Q.shape[1]) for e2,Q2 in D5]
        print(f'L0H{j} eig %.2f@%d'%(abs(e),np.degrees(np.angle(e))),'overlap with H1 planes',ov,'H5 planes',ov5,'H4plane %.2f'%(np.linalg.norm(Q.T@P4)**2/Q.shape[1]))
print('H1 planes:',['%.2f@%d'%(abs(e),np.degrees(np.angle(e))) for e,Q in D1])
print('H5 planes:',['%.2f@%d'%(abs(e),np.degrees(np.angle(e))) for e,Q in D5])
```