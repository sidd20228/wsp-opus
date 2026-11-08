```python
import numpy as np
hd=64
D=[W[f'ln1_w.{l}'] for l in range(4)]
B=[W[f'ln1_b.{l}'] for l in range(4)]
sl=lambda h: slice(h*hd,(h+1)*hd)
def rsvd(M):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt,S
V1,s1=rsvd(W['W_v.0'][sl(1)])
Tx,_=np.linalg.qr(D[0][:,None]*V1[:27].T)
gs=[]
for l,h in [(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(2,2),(2,3)]:
    qb=W[f'W_q.{l}'][sl(h)]@B[l]
    g=((W[f'W_k.{l}'][sl(h)]*D[l])@Tx).T@qb
    gs.append(g/np.linalg.norm(g))
gm=np.mean(gs,0); gm/=np.linalg.norm(gm)
# H1 V functional directions in Tx coords
F=[]
for i in range(27):
    f=D[0]*V1[i]; F.append(Tx.T@f)
F=np.array(F)
Fn=F/np.linalg.norm(F,axis=1,keepdims=True)
print('cos(g, H1 V dir i):',np.round(Fn@gm,2))
# also H5 V dirs
V5,s5=rsvd(W['W_v.0'][sl(5)])
F5=np.array([Tx.T@(D[0]*V5[i]) for i in range(27)]); F5n=F5/np.linalg.norm(F5,axis=1,keepdims=True)
print('cos(g, H5 V dir i):',np.round(F5n@gm,2))
# MLP0 alive top read dirs
Fc=W['W_fc.0']*W['ln2_w.0']; alive=np.linalg.norm(Fc,axis=1)>0.6
Vm,sm=rsvd(Fc[alive]@Tx)
print('MLP0 tok-read sv',np.round(sm[:10],1),' cos(g, MLP0 tok dirs):',np.round(Vm[:10]@gm,2))
```