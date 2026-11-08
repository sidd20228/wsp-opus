```python
w=W['ln1_w.0']
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=head(0,2)
u,s,vt=np.linalg.svd(Wq.T@Wk)
P=np.hstack([u[:,:20],vt[:20].T]); P,_=np.linalg.qr(P)
pdiag=(P**2).sum(1)   # how much each coordinate is in P
uv,sv,vvt=np.linalg.svd(Wo@Wv)
T=vvt[:24].T
tdiag=(T**2).sum(1)
order=np.argsort(-w)
print('coord w | Pweight | Tweight for top-60 w dims')
for i in order[:60]:
    print(i, round(float(w[i]),2), round(float(pdiag[i]),3), round(float(tdiag[i]),3))
print('sum P weight over big dims', pdiag[w>0.5].sum(), 'of', pdiag.sum())
print('sum T weight over big dims', tdiag[w>0.5].sum(), 'of', tdiag.sum())
```