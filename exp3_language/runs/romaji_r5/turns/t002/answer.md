```python
import numpy as np
np.set_printoptions(linewidth=200)
d=384; hd=64
def heads(l):
    w=W[f'ln1_w.{l}']; res=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        res.append((W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]))
    return res
H={l:heads(l) for l in range(4)}
A={(l,j):H[l][j][0].T@H[l][j][1] for l in range(4) for j in range(6)}
OV={(l,j):H[l][j][3]@H[l][j][2] for l in range(4) for j in range(6)}
print("singular values of QK (A) per head: top 8, and 48*smax (max possible score for unit-variance LN input)")
for L in range(4):
    for h in range(6):
        s=np.linalg.svd(A[(L,h)],compute_uv=False)
        so=np.linalg.svd(OV[(L,h)],compute_uv=False)
        print(f'L{L}H{h} QKsv',np.round(s[:8],2),' maxscore %.1f'%(48*s[0]),' OVsv',np.round(so[:6],2))
# bias contributions: query/key from ln bias
print("\nLN-bias driven terms: |k_bias|, |q_bias|, and key-bias term norm  q_b^T k  (query-independent part)")
for L in range(4):
    b=W[f'ln1_b.{L}']
    for h in range(6):
        s=slice(h*hd,(h+1)*hd)
        qb=W[f'W_q.{L}'][s]@b; kb=W[f'W_k.{L}'][s]@b
        q,k,v,o=H[L][h]
        kterm=qb@k   # score contribution from query bias, depends on key: direction in residual
        print(f'L{L}H{h} |qb|={np.linalg.norm(qb):.2f} |kb|={np.linalg.norm(kb):.2f} |qb^T k|={np.linalg.norm(kterm):.2f}  max key-only score ~ {19.6*np.linalg.norm(kterm)/8:.1f}')
```