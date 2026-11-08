<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def orth(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:r]
q,k,v,o=hw(0,3); A3=q.T@k; u3,s3,v3=np.linalg.svd(A3)
H3plane=orth(np.hstack([u3[:,:2],v3[:2].T]),3)
print('H3 q/k plane sv', s3[:3]*48)
for i in [0,1,4,5]:
    q,k,v,o=hw(0,i); U,S,Vt=np.linalg.svd(o@v)
    print(i,'OV in-dir overlap with H3 plane:',[round(float(np.linalg.norm(H3plane.T@Vt[r])**2),2) for r in range(3)], 'S',S[:3].round(2))
# also: H2 QK (prev) top dirs vs H3 plane
q,k,v,o=hw(0,2); A2=q.T@k; u2,s2,v2=np.linalg.svd(A2)
print('H3 plane energy in H2 q-space top20:', np.linalg.svd(H3plane.T@u2[:,:20],compute_uv=False).round(2))
# L1 query side reads: overlap of L1 Q's top dirs with H3 plane
for j in range(6):
    q,k,v,o=hw(1,j); QK=q.T@k; U,S,Vt=np.linalg.svd(QK)
    print('L1H%d top2 q-dirs energy in H3plane'%j, [round(float(np.linalg.norm(H3plane.T@U[:,r])**2),2) for r in range(3)], ' k-dirs',[round(float(np.linalg.norm(H3plane.T@Vt[r])**2),2) for r in range(3)])
```

````