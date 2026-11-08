```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=head(0,2)
OV02=Wo@Wv
for (l,j) in [(3,1),(3,0),(1,2),(1,3),(2,1)]:
    a,b,c,d=head(l,j); QK=a.T@b
    N=QK@OV02
    ev=np.linalg.eigvals(N); o=np.argsort(-ev.real)
    print(f'L{l}H{j} top real eig', np.round(ev[o][:34].real,2))
    print('   most negative', np.round(np.sort(ev.real)[:6],2), ' max |imag| among top30 %.2f'%np.abs(ev[o][:30].imag).max())
```