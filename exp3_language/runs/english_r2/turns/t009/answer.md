```python
def head(l,j):
    s=slice(64*j,64*j+64)
    w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
for j in [0,1,4,5,3,2]:
    Wq,Wk,Wv,Wo=head(0,j)
    OVm=Wo@Wv; QKm=Wq.T@Wk
    u,s,vt=np.linalg.svd(OVm); 
    uq,sq,vq=np.linalg.svd(QKm)
    def top(v,n=8):
        idx=np.argsort(-np.abs(v))[:n]; return ' '.join(f'{i}:{v[i]:+.2f}' for i in idx)
    print(f'H{j} OVin  ', top(vt[0]), ' PR %.1f'%(1/np.sum(vt[0]**4)))
    print(f'H{j} OVout ', top(u[:,0]), ' PR %.1f'%(1/np.sum(u[:,0]**4)))
    print(f'H{j} QKq   ', top(uq[:,0]), ' PR %.1f'%(1/np.sum(uq[:,0]**4)))
    print(f'H{j} QKk   ', top(vq[0]), ' PR %.1f'%(1/np.sum(vq[0]**4)))
```