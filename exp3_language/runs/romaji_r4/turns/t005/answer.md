```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
OVs=[head(0,j)[3]@head(0,j)[2] for j in range(H)]
U,s,_=np.linalg.svd(np.hstack([OVs[1],OVs[2],OVs[4],OVs[5]]),full_matrices=False)
w=U[:,0]; r=w@OVs[5]; r/=fro(r)
uu={};vv={}
for j in range(H):
    Q,K,V,O=head(0,j);u,ss,vt=np.linalg.svd(Q.T@K);uu[j]=u[:,0]*np.sign(u[:,0]@r+1e-9);vv[j]=vt[0]*np.sign(u[:,0]@r+1e-9)
vecs={'r':r}
for j in [0,1,2,4,5]: vecs[f'u{j}']=uu[j];vecs[f'v{j}']=vv[j]
# second OV direction of L0H5, L0H1
for j in [1,2,5]:
    uo,so,vo=np.linalg.svd(OVs[j]);vecs[f'r2_{j}']=vo[1];vecs[f'w2_{j}']=uo[:,1]
vecs['w']=w
keys=list(vecs)
print('      '+' '.join('%6s'%k for k in keys))
for a in keys: print('%6s'%a+' '.join('%+6.2f'%(vecs[a]@vecs[b]) for b in keys))
# L0H3 rotation planes: span of top 16 complex eigvecs
Q,K,V,O=head(0,3);M=Q.T@K
ev,evec=np.linalg.eig(M);idx=np.argsort(-np.abs(ev))[:20]
B=np.hstack([evec[:,idx].real,evec[:,idx].imag]);Bq,_=np.linalg.qr(B)
print('fraction of each vec in L0H3 top-20 eig planes:',' '.join('%s %.2f'%(k,fro(Bq.T@vecs[k])**2) for k in keys))
```