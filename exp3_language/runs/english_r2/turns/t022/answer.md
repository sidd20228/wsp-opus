```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=head(0,2)
u,s,vt=np.linalg.svd(Wq.T@Wk)
P,_=np.linalg.qr(np.hstack([u[:,:20],vt[:20].T]))
uv,sv,vvt=np.linalg.svd(Wo@Wv); T=vvt[:24].T; O2=uv[:,:24]
S=[222,78,360,103,72,254,25,197,273,136,307,249,292,226,302,130]
mask=np.ones(384); mask[S]=0
# L0 broad head outputs (rank1)
outs=[]
for j in [0,1,3,4,5]:
    a,b,c,d=head(0,j); uu,ss,vv=np.linalg.svd(d@c); outs.append(uu[:,:2])
B0=np.linalg.qr(np.hstack(outs))[0]
def cov(v,B): return np.linalg.norm(B.T@v)**2
for l in [1,2,3]:
  for j in range(6):
    Wq,Wk,Wv,Wo=head(l,j); QK=(Wq.T@Wk)*np.outer(mask,mask)
    uq,sq,vq=np.linalg.svd(QK)
    print(f'L{l}H{j} masked sv', sq[:6].round(2))
    for r in range(3):
        print(f'   r{r}: q in T {cov(uq[:,r],T):.2f} O2 {cov(uq[:,r],O2):.2f} P {cov(uq[:,r],P):.2f} B0 {cov(uq[:,r],B0):.2f} | k in T {cov(vq[r],T):.2f} O2 {cov(vq[r],O2):.2f} P {cov(vq[r],P):.2f} B0 {cov(vq[r],B0):.2f} | q.k {uq[:,r]@vq[r]:+.2f}')
```