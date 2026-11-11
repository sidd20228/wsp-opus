```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
sf=np.linalg.svd(W['W_fc.0'],compute_uv=False)
print('fc0 sv 40..160 step4',np.round(sf[40:160:4],2))
for nm in ['W_q.0','W_k.0','W_v.0']:
    s=np.linalg.svd(W[nm],compute_uv=False);print(nm,np.round(s[0:120:6],2))
Pemb=np.linalg.svd(W['W_fc.0'],full_matrices=False)[2][:60].T
def top(M,k): return np.linalg.svd(M,full_matrices=False)[0][:,:k]
srcs={'E':Pemb}
for j in range(H): srcs[f'a{j}']=top(head(0,j)[3]@head(0,j)[2],8)
srcs['M0']=top(W['W_proj.0'],24)
for l in [1,2]:
    for j in range(H):
        Q,K,V,O=head(l,j);u,s,vt=np.linalg.svd(Q.T@K)
        out=[]
        for i in range(3):
            for side,vec in [('q',u[:,i]),('k',vt[i])]:
                sc=sorted([(k,fro(P.T@vec)**2) for k,P in srcs.items()],key=lambda t:-t[1])[:3]
                out.append(f'{side}{i}('+','.join('%s%.2f'%t for t in sc)+')')
        print(f'L{l}H{j} s=%s '%np.round(s[:3],1)+' '.join(out))
```