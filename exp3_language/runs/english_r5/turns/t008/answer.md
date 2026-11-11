```python
import numpy as np
D=W['ln1_w.0']
def rows(m,j,l=0): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
def top(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k]
Vb=top(np.vstack([rows('v',j) for j in [0,1,4,5]]),3)
Q2=top(rows('q',2),20); K2=top(rows('k',2),20)
Q3=top(rows('q',3),3); K3=top(rows('k',3),3)
def ov(A,B):
    return np.linalg.svd(A@B.T,compute_uv=False)
print('Vb vs Q2',ov(Vb,Q2).round(2))
print('Vb vs K2',ov(Vb,K2).round(2))
print('Q2 vs K2',ov(Q2,K2).round(2))
print('Vb vs Q3',ov(Vb,Q3).round(2),'Vb vs K3',ov(Vb,K3).round(2))
print('Q3 vs K3',ov(Q3,K3).round(2))
print('Q2 vs K3',ov(Q2,K3).round(2),'K2 vs Q3',ov(K2,Q3).round(2))
# each broad head's top directions
for j in [0,1,4,5]:
    u,s,vt=np.linalg.svd(rows('v',j),full_matrices=False)
    print(j,s[:6].round(2), 'overlap with Vb top3', np.linalg.norm(vt[:3]@Vb.T,axis=1).round(2))
# also the bias b in which directions
b=W['ln1_b.0']
print('|b|',np.linalg.norm(b))
```