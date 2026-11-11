```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
sf=np.linalg.svd(W['W_fc.0'],compute_uv=False)
print('fc0 sv top45',np.round(sf[:45],2))
Pemb=np.linalg.svd(W['W_fc.0'],full_matrices=False)[2][:40].T
def idc(M): return np.trace(M)/fro(M)/np.sqrt(M.shape[0])
print('per head: QK s1, s1/s2, same-char idcorr(emb), OV s1')
for l in range(4):
    row=[]
    for j in range(H):
        Q,K,V,O=head(l,j);M=Q.T@K;s=np.linalg.svd(M,compute_uv=False);sv=np.linalg.svd(O@V,compute_uv=False)
        row.append('H%d %.2f %.1f %+.2f %.2f'%(j,s[0],s[0]/s[1],idc(Pemb.T@M@Pemb),sv[0]))
    print('L%d | '%l+' | '.join(row))
rng=np.random.default_rng(0)
R=np.linalg.qr(rng.normal(size=(384,384)))[0]
print('K-comp with L0 heads (ratio to random-rotation baseline) and induction idcorr (q=emb, k-source=emb via L0 head)')
for l in range(1,4):
    for j in range(H):
        Q,K,V,O=head(l,j);QK=Q.T@K
        kr=[];ic=[]
        for j1 in range(H):
            Q1,K1,V1,O1=head(0,j1);OV=O1@V1
            k=fro(K@OV)/(fro(K)*fro(OV));b=fro(K@R@OV)/(fro(K)*fro(OV))
            kr.append(k/b);ic.append(idc(Pemb.T@QK@OV@Pemb))
        print(f'L{l}H{j} kc:',' '.join('%.1f'%x for x in kr),' ind:',' '.join('%+.2f'%x for x in ic))
```