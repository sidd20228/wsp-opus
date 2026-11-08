```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=head(0,2)
u,s,vt=np.linalg.svd(Wq.T@Wk)
P,_=np.linalg.qr(np.hstack([u[:,:20],vt[:20].T]))
uv,sv,vvt=np.linalg.svd(Wo@Wv); T=vvt[:24].T
O2=uv[:,:24]
S=[222,78,360,103,72,254,25,197,273,136,307,249,292,226,302,130]
Sb=np.eye(384)[:,S]
def f(M,B): return np.linalg.norm(M@B)**2/np.linalg.norm(M)**2
def fq(M,B): return np.linalg.norm(B.T@M)**2/np.linalg.norm(M)**2
print('baselines P %.3f T %.3f O2 %.3f S %.3f'%(40/384,24/384,24/384,16/384))
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=head(l,j); OV=Wo@Wv; QK=Wq.T@Wk
        print(f'L{l}H{j} OVin: P {f(OV,P):.2f} T {f(OV,T):.2f} O2 {f(OV,O2):.2f} S {f(OV,Sb):.2f} | Qside: T {fq(QK,T):.2f} O2 {fq(QK,O2):.2f} | Kside: T {f(QK,T):.2f} O2 {f(QK,O2):.2f}')
```