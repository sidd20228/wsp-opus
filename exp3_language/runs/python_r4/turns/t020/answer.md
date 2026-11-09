```python
import numpy as np
hd=64
def OVl(l,j):
    lw=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw)
k=24
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
def outsp(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k]
S1=outsp(OVl(0,1),20); S5=outsp(OVl(0,5),20); S2=outsp(OVl(0,2),10); S3=outsp(OVl(0,3),10); S4=outsp(OVl(0,4)+OVl(0,0),3)
def frac(A,S): return (np.linalg.norm(A@S)**2/np.linalg.norm(A)**2)/(S.shape[1]/384)
for l in range(4):
    A=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    print('MLP',l,'reads: B %.1f S1 %.1f S5 %.1f S2 %.1f S3 %.1f pos %.1f'%(frac(A,B),frac(A,S1),frac(A,S5),frac(A,S2),frac(A,S3),frac(A,S4)))
# per-neuron: fraction of each neuron's weight energy in B, S1, S5
A=W['W_fc.0']*W['ln2_w.0']
nb=np.linalg.norm(A@B,axis=1)**2; n1=np.linalg.norm(A@S1,axis=1)**2; n5=np.linalg.norm(A@S5,axis=1)**2; nt=np.linalg.norm(A,axis=1)**2
big=nt>np.percentile(nt,50)
print('neurons (top half by norm): mean frac B %.2f S1 %.2f S5 %.2f'%((nb/nt)[big].mean(),(n1/nt)[big].mean(),(n5/nt)[big].mean()))
print('corr of B-frac and S1+S5-frac', np.corrcoef((nb/nt)[big],((n1+n5)/nt)[big])[0,1])
both=((nb/nt)>0.15)&(((n1+n5)/nt)>0.15)
print('neurons reading both B and prev subspaces (>0.15 each):',both.sum(),'of',len(nt))
```