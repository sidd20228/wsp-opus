```python
def head(l,j):
    s=slice(64*j,64*j+64)
    w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
for (l,j) in [(3,0),(3,1),(3,4),(2,1),(0,3),(0,2)]:
    Wq,Wk,Wv,Wo=head(l,j); M=Wq.T@Wk
    d=np.diag(M); idx=np.argsort(-np.abs(d))[:12]
    print(f'L{l}H{j} trace {d.sum():.2f} fro {np.linalg.norm(M):.2f} top diag', [(i,round(float(d[i]),2)) for i in idx])
    print('   trace without top12 %.2f'%(d.sum()-d[idx].sum()), ' median diag %.4f'%np.median(d), 'frac positive diag %.2f'%(d>0).mean())
    ev=np.linalg.eigvalsh((M+M.T)/2); print('   sym eig top',ev[-6:].round(2),'bottom',ev[:4].round(2))
```