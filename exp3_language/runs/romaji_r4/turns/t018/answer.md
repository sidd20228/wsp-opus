```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
OV5=head(0,5)[3]@head(0,5)[2];U5,s5,V5=np.linalg.svd(OV5)
r=V5[0];w=U5[:,0]
info={}
for j in [1,2,4,5]:
    Q,K,V,O=head(0,j);OV=O@V;uo,so,vo=np.linalg.svd(OV);u,s,vt=np.linalg.svd(Q.T@K)
    info[j]=(uo[:,:2],vo[:2],u[:,:2],vt[:2])
    print('L0H%d OV comps: in1.r %+.2f in2.r %+.2f | out1.w %+.2f out2.w %+.2f | in1.keyv %+.2f in1.queryu %+.2f | in2.keyv %+.2f'%(j,vo[0]@r,vo[1]@r,uo[:,0]@w,uo[:,1]@w,vo[0]@vt[0],vo[0]@u[:,0],vo[1]@vt[0]))
print('L0H5 OV2 in . r? %.2f; out2 . w %.2f'%(V5[1]@r,U5[:,1]@w))
# cross: out1 of heads 1,2,4 alignment
for a in [1,2,4,5]:
    print(a,' '.join('%+.2f'%(info[a][0][:,0]@info[b][0][:,0]) for b in [1,2,4,5]),' | in1:',' '.join('%+.2f'%(info[a][1][0]@info[b][1][0]) for b in [1,2,4,5]))
# who reads out1 of L0H1 (o1) in later layers (q/k/v relative to spectral norm)
o1=info[1][0][:,0]
for l in range(1,4):
    print('L%d reads L0H1-out1:'%l,' '.join('H%d q%.2f k%.2f v%.2f'%(j,fro(head(l,j)[0]@o1)/np.linalg.norm(head(l,j)[0],2),fro(head(l,j)[1]@o1)/np.linalg.norm(head(l,j)[1],2),fro(head(l,j)[2]@o1)/np.linalg.norm(head(l,j)[2],2)) for j in range(H)))
```