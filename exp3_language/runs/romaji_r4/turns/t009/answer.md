```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Q3,K3,V3,O3=head(0,3);OV3=O3@V3;M3=Q3.T@K3
ev,evec=np.linalg.eig(M3);order=np.argsort(-np.abs(ev))[:20]
Bp,_=np.linalg.qr(np.hstack([evec[:,order].real,evec[:,order].imag]));Pr=np.eye(384)-Bp@Bp.T
uo,so,vo=np.linalg.svd(OV3@Pr);T=vo[:22].T
vk={}
for j in [1,2,4]:
    Q,K,V,O=head(0,j);u,s,vt=np.linalg.svd(Q.T@K);vk[j]=vt[0]
    print('L0H%d key dir v: frac in T %.2f, in pos planes %.2f'%(j,fro(T.T@vt[0])**2,fro(Bp.T@vt[0])**2))
F=W['W_fc.0']*W['ln2_w.0']
uf,sf,vf=np.linalg.svd(F,full_matrices=False)
for i in range(4):
    print('fc0 dir %d sv %.1f: in T %.2f, in pos %.2f, cos with v1 %.2f v2 %.2f v4 %.2f'%(i,sf[i],fro(T.T@vf[i])**2,fro(Bp.T@vf[i])**2,vf[i]@vk[1],vf[i]@vk[2],vf[i]@vk[4]))
# token-space spectrum of fc0 reads
s=np.linalg.svd(F@T,compute_uv=False);print('fc0 read in T sv',np.round(s,1))
for l in range(1,3):
    for j in range(H):
        Q,K,V,O=head(l,j);Mt=T.T@Q.T@K@T;S=(Mt+Mt.T)/2;e=np.linalg.eigvalsh(S)
        # dominant eigvec alignment with v (special token)
        w_,V_=np.linalg.eigh(S);d=T@V_[:,0]
        print('L%dH%d tok-tok sym eig low %s high %s ; most-neg eigvec cos v1 %.2f v2 %.2f'%(l,j,np.round(e[:3],2),np.round(e[-3:],2),d@vk[1],d@vk[2]))
```