```python
def head(l,j):
    s=slice(64*j,64*j+64)
    w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
QK={};OV={}
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=head(l,j); QK[l,j]=Wq.T@Wk; OV[l,j]=Wo@Wv
print('OV02 sv', np.linalg.svd(OV[0,2],compute_uv=False)[:40].round(2))
print('Wv02 sv', np.linalg.svd(head(0,2)[2],compute_uv=False)[:40].round(2))
rng=np.random.default_rng(0)
for j in range(6):
    C=QK[1,j]@OV[0,2]
    ev=np.linalg.eigvals(C); 
    o=np.argsort(-np.abs(ev))[:10]
    print(f'L1H{j}: top eig', np.round(ev[o],2), ' trace %.2f  sum|ev| %.2f'%(np.trace(C).real, np.abs(ev).sum()))
```