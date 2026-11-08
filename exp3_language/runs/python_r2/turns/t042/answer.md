```python
import numpy as np
hd=64
D=[W[f'ln1_w.{l}'] for l in range(4)]
sl=lambda h: slice(h*hd,(h+1)*hd)
def rsvd(M):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt,S
OV=lambda l,h: W[f'W_o.{l}'][:,sl(h)]@(W[f'W_v.{l}'][sl(h)]*D[l])
V1,_=rsvd(W['W_v.0'][sl(1)])
Tx,_=np.linalg.qr(D[0][:,None]*V1[:27].T)
Fc=W['W_fc.0']*W['ln2_w.0']
alive=np.linalg.norm(Fc,axis=1)>0.6
Fa=Fc[alive]
A=Fa@Tx                          # current-token functional
B1=Fa@OV(0,1)@Tx                 # prev token via H1
B5=Fa@OV(0,5)@Tx
for nm,Bm in [('H1',B1),('H5',B5),('H1+H5',B1+B5)]:
    an=A/np.linalg.norm(A,axis=1,keepdims=True); bn=Bm/np.linalg.norm(Bm,axis=1,keepdims=True)
    c=(an*bn).sum(1)
    rng=np.random.default_rng(0); perm=rng.permutation(len(c))
    cr=(an*bn[perm]).sum(1)
    print(nm,'cos(a,b) pct',np.round(np.percentile(c,[5,25,50,75,95]),2),' null',np.round(np.percentile(cr,[5,25,50,75,95]),2),
          ' frac>0.5: %.3f null %.3f ; frac<-0.5: %.3f null %.3f'%((c>0.5).mean(),(cr>0.5).mean(),(c<-0.5).mean(),(cr<-0.5).mean()))
    print('   norms a med %.2f b med %.2f'%(np.median(np.linalg.norm(A,axis=1)),np.median(np.linalg.norm(Bm,axis=1))))
```