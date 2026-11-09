```python
import numpy as np
np.set_printoptions(linewidth=220)
d=384; hd=64; w0=W['ln1_w.0']
Wv=W['W_v.0']*w0; Wq=W['W_q.0']*w0; Wk=W['W_k.0']*w0
print('stacked V.0 svals:',np.round(np.linalg.svd(Wv,compute_uv=False)[:60],2))
print('stacked Q.0 svals:',np.round(np.linalg.svd(Wq,compute_uv=False)[:60],2))
print('stacked K.0 svals:',np.round(np.linalg.svd(Wk,compute_uv=False)[:60],2))
WO=W['W_o.0']@Wv
print('full OV.0 svals:',np.round(np.linalg.svd(WO,compute_uv=False)[:60],2))
# exclude H5 (uniform)
idx=np.r_[0:320]
print('V.0 without H5:',np.round(np.linalg.svd(Wv[idx],compute_uv=False)[:60],2))
def heads(l):
    w=W[f'ln1_w.{l}']; res=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        res.append((W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]))
    return res
H={l:heads(l) for l in range(4)}
q,k,v,o=H[3][3]
M=q.T@k@(H[0][3][3]@H[0][3][2])
ev=np.linalg.eigvals(M); ev=ev[np.argsort(-np.abs(ev))][:50]
print('L3H3 A@OV(L0H3) eig:',' '.join('%.2f@%.0f'%(abs(z),np.degrees(np.angle(z))) for z in ev if np.angle(z)>=-1e-9))
ev=np.linalg.eigvals(q.T@k); ev=ev[np.argsort(-np.abs(ev))][:64]
print('L3H3 A eig:',' '.join('%.2f@%.0f'%(abs(z),np.degrees(np.angle(z))) for z in ev if np.angle(z)>=-1e-9))
```