```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
a,b,c,d=head(0,2); OV02=d@c
S=[222,78,360,103,72,254,25,197,273,136,307,249,292,226,302,130]
mask=np.ones(384); mask[S]=0
u,s,vt=np.linalg.svd(c,full_matrices=False)
def sym(M): return (M+M.T)/2
a3,b3,_,_=head(3,1); QK31=a3.T@b3
QK1=[(head(1,j)[0].T@head(1,j)[1])*np.outer(mask,mask) for j in range(6)]
for k in [24,32,40,48,64]:
    T=vt[:k].T
    ind=np.linalg.eigvalsh(sym(T.T@QK31@OV02@T))[::-1]
    dup=np.linalg.eigvalsh(-sym(T.T@sum(QK1)@T))[::-1]
    print(k,'ind pos eig >5%%max: %d, >2%%: %d, neg<-5%%: %d'%((ind>0.05*ind[0]).sum(),(ind>0.02*ind[0]).sum(),(ind<-0.05*ind[0]).sum()),
          '| dup(sum L1) >5%%: %d >2%%: %d neg: %d'%((dup>0.05*dup[0]).sum(),(dup>0.02*dup[0]).sum(),(dup<-0.05*dup[0]).sum()))
    print('   ind', ind[:30].round(3))
    print('   dup', dup[:30].round(3), dup[-4:].round(3))
```