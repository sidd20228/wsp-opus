```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W['ln1_w.0']; H=w0>0.45
def hq(l,j):
    s=slice(64*j,64*j+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
U,S,Vt=np.linalg.svd(W['W_v.0'][256:320]*w0,full_matrices=False); D=Vt[:2].T
O4=np.linalg.svd(hq(0,4)[3]@hq(0,4)[2])[0][:,:3]; O0=np.linalg.svd(hq(0,0)[3]@hq(0,0)[2])[0][:,:2]
Hb=np.eye(384)[:,H]
Pp=np.linalg.svd(np.concatenate([Hb,O4,O0],1),full_matrices=False)[0][:,:75]
for L in [1,2,3]:
    for h in range(6):
        q,k,v,o=hq(L,h); M=q.T@k
        Mp=Pp.T@M@Pp
        ev,V=np.linalg.eig(Mp)
        o_=np.argsort(-np.abs(ev))
        out=[]
        c=0
        for i in o_:
            e=ev[i]
            if e.imag<-1e-8: continue
            vv=Pp@V[:,i]
            B=np.stack([vv.real,vv.imag],1) if abs(e.imag)>1e-8 else vv.real[:,None]
            Qb,_=np.linalg.qr(B)
            dov=np.linalg.norm(D.T@Qb); o4=np.linalg.norm(O4.T@Qb)
            out.append(f"{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}(D{dov:.2f},O4{o4:.2f})")
            c+=1
            if c>=5: break
        print(f'L{L}H{h} |Mpp|={np.linalg.norm(Mp):.1f}', ' '.join(out))
```