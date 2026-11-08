```python
def head(l,j):
    s=slice(64*j,64*j+64)
    w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
dims=[222,360,78,72,103,254,25]
for l in range(4):
    print(l,'ln1_w',W[f'ln1_w.{l}'][dims].round(2),'ln1_b',W[f'ln1_b.{l}'][dims].round(2))
    print(l,'ln2_w',W[f'ln2_w.{l}'][dims].round(2),'ln2_b',W[f'ln2_b.{l}'][dims].round(2))
for j in range(6):
    Wq,Wk,Wv,Wo=head(1,j)
    QKm=Wq.T@Wk
    print('L1H',j); print(QKm[np.ix_(dims,dims)].round(2))
    print(' q-col norms', np.linalg.norm(Wq[:,dims],axis=0).round(2), ' k-col norms', np.linalg.norm(Wk[:,dims],axis=0).round(2), 'median col norm q %.2f k %.2f'%(np.median(np.linalg.norm(Wq,axis=0)),np.median(np.linalg.norm(Wk,axis=0))))
```