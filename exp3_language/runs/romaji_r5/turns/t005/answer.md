```python
import numpy as np, pandas as pd
np.set_printoptions(linewidth=220); pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
d=384; hd=64
w0=W['ln1_w.0']
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
def pc(U1,U2): return np.linalg.svd(U1@U2.T,compute_uv=False)
subs={}
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    subs[f'q{j}']=topRows(W['W_q.0'][s]*w0,6); subs[f'k{j}']=topRows(W['W_k.0'][s]*w0,6); subs[f'v{j}']=topRows(W['W_v.0'][s]*w0,6)
subs['fc']=topRows(W['W_fc.0']*W['ln2_w.0'],12)
names=list(subs)
M=np.array([[np.mean(pc(subs[a],subs[b])**2) for b in names] for a in names])
print(pd.DataFrame(np.round(M,2),index=names,columns=names).to_string())
print("ln1_w.0 sorted:",np.round(np.sort(w0)[::-1][:40],2))
print("singular values of v per L0 head:")
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    print(j,np.round(np.linalg.svd(W['W_v.0'][s]*w0,compute_uv=False)[:12],2), ' q',np.round(np.linalg.svd(W['W_q.0'][s]*w0,compute_uv=False)[:6],2),' k',np.round(np.linalg.svd(W['W_k.0'][s]*w0,compute_uv=False)[:6],2))
```