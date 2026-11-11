```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Q3,K3,V3,O3=head(0,3);OV3=O3@V3
uo,so,vo=np.linalg.svd(OV3)
print('L0H3 OV sv 18..64',np.round(so[18:64],2))
T=vo[:26].T; P3=uo[:,:20]
def idc(M): return np.trace(M)/fro(M)/np.sqrt(M.shape[0])
# what do L1/L2 heads' V read: token space T, L0H3 output P3, w?
OVs=[head(0,j)[3]@head(0,j)[2] for j in range(H)]
U,s,_=np.linalg.svd(np.hstack([OVs[1],OVs[2],OVs[4],OVs[5]]),full_matrices=False);w=U[:,0]
for l in range(1,4):
    print('L%d V reads:'%l,' '.join('H%d T%.2f P3 %.2f w%.3f'%(j,fro(head(l,j)[2]@T)**2/fro(head(l,j)[2])**2,fro(head(l,j)[2]@P3)**2/fro(head(l,j)[2])**2,fro(head(l,j)[2]@w)**2/fro(head(l,j)[2])**2) for j in range(H)))
print('random expectation T %.3f P3 %.3f'%(26/384,20/384))
# chained matching: q=token(T), key-source = token via earlier head OV (l1,j1)
for l in range(2,4):
    for j in range(H):
        Q,K,V,O=head(l,j);QK=Q.T@K
        out=[]
        for l1 in range(1,l):
            for j1 in range(H):
                OV=head(l1,j1)[3]@head(l1,j1)[2]
                M=T.T@QK@OV@T;out.append('%d%d:%+.2f'%(l1,j1,idc(M)))
        print(f'L{l}H{j}',' '.join(out))
```