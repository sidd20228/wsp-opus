```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
OVs=[head(0,j)[3]@head(0,j)[2] for j in range(H)]
for j in range(H):
    s=np.linalg.svd(OVs[j],compute_uv=False);print('L0H%d OV sv'%j,np.round(s[:8],2))
U,s,_=np.linalg.svd(np.hstack([OVs[1],OVs[2],OVs[4],OVs[5]]),full_matrices=False)
print('concat OV(1,2,4,5) sv',np.round(s[:8],2))
w=U[:,0]
print('|w^T OV_j| per head:',np.round([fro(w@OV) for OV in OVs],2))
R=[w@OV/ (fro(w@OV)+1e-9) for OV in OVs]
print('cos between input dirs r_j producing w:')
for a in range(H): print(' '.join('%+.2f'%(R[a]@R[b]) for b in range(H)))
# is the input dir aligned with ln1_b.0 (LN bias, a constant)? 
b=W['ln1_b.0'];print('cos(r_j, ln1_b.0):',np.round([R[j]@b/fro(b) for j in range(H)],2),' |ln1_b.0|=%.2f'%fro(b))
# L0 QK: top singular vectors, q-side & k-side, cos with r (shared input dir)
r=R[5]
for j in range(H):
    Q,K,V,O=head(0,j);u,ss,vt=np.linalg.svd(Q.T@K)
    print('L0H%d QK s'%j,np.round(ss[:5],2),'cos(u_i,r)',np.round([u[:,i]@r for i in range(4)],2),'cos(v_i,r)',np.round([vt[i]@r for i in range(4)],2),'|Qb|.|K|: %.2f'%fro((Q@b)@K))
# later layer readers of w
for l in range(1,4):
    print('L%d'%l,' '.join('H%d q%.2f k%.2f v%.2f'%(j,fro(head(l,j)[0]@w)/np.linalg.norm(head(l,j)[0],2),fro(head(l,j)[1]@w)/np.linalg.norm(head(l,j)[1],2),fro(head(l,j)[2]@w)/np.linalg.norm(head(l,j)[2],2)) for j in range(H)))
```