```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Up,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
mats={'L3H3':head(3,3)[3]@head(3,3)[2],'Wproj2':W['W_proj.2'],'Wproj1':W['W_proj.1'],'L3H0':head(3,0)[3]@head(3,0)[2],'L2all_o':W['W_o.2']}
for nm,A in mats.items():
    e=np.array([fro(Up[:,i]@A) for i in range(384)])/ (fro(A)/np.sqrt(384))
    print('%-8s'%nm,'dirs0-20 mean %.2f | 21-27:'%e[:21].mean(),np.round(e[21:28],2),'| 28-40 mean %.2f | 40-100 %.2f | 100+ %.2f'%(e[28:40].mean(),e[40:100].mean(),e[100:].mean()))
print('W_proj.3 sv 18..34',np.round(sp[18:34],2))
```