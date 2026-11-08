```python
import numpy as np
hd=64
D=[W[f'ln1_w.{l}'] for l in range(4)]
sl=lambda h: slice(h*hd,(h+1)*hd)
def rsvd(M):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt,S
def colsp(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k]
OV=lambda l,h: W[f'W_o.{l}'][:,sl(h)]@(W[f'W_v.{l}'][sl(h)]*D[l])
V1,_=rsvd(W['W_v.0'][sl(1)])
Tx,_=np.linalg.qr(D[0][:,None]*V1[:27].T)
O04=colsp(np.hstack([OV(0,0),OV(0,4)]),3)
O5=colsp(OV(0,5),23); O1=colsp(OV(0,1),27)
Fc=W['W_fc.0']*W['ln2_w.0']
Vm,sm=rsvd(Fc)
for i in range(6):
    v=Vm[i]
    print(i,'sv %.1f'%sm[i],'T %.2f O04 %.2f O1 %.2f O5 %.2f'%(np.linalg.norm(Tx.T@v),np.linalg.norm(O04.T@v),np.linalg.norm(O1.T@v),np.linalg.norm(O5.T@v)))
# W_proj.0 output: where does MLP0 write? overlap with T, O-spaces
U0,S0,_=np.linalg.svd(W['W_proj.0'],full_matrices=False)
for i in range(4): print('proj0 out dir',i,'sv %.1f'%S0[i],'T %.2f O04 %.2f'%(np.linalg.norm(Tx.T@U0[:,i]),np.linalg.norm(O04.T@U0[:,i])))
# ln2_b.0 and ln1_b.1 alignment
for nm in ['ln2_b.0','ln1_b.1']:
    b=W[nm]/np.linalg.norm(W[nm]); print(nm,'T %.2f O04 %.2f'%(np.linalg.norm(Tx.T@b),np.linalg.norm(O04.T@b)))
```