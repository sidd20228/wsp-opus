```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w1 = W['ln1_w.0']
def rows(m, r): return np.linalg.svd(m)[2][:r]
st = np.vstack([W['W_q.0']*w1, W['W_k.0']*w1, W['W_v.0']*w1])
U,s,Vt = np.linalg.svd(st, full_matrices=False)
print('attn0 read sv', s[:90])
q3 = W['W_q.0'][192:256]*w1; k3 = W['W_k.0'][192:256]*w1; v3 = W['W_v.0'][192:256]*w1
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 20)
Wfc = W['W_fc.0']*W['ln2_w.0']
F = rows(Wfc, 45)
for i in range(70):
    d = Vt[i]
    print(i, f'{s[i]:.2f}', 'S %.2f F %.2f V3 %.2f' % tuple(np.linalg.norm(X@d)**2 for X in (S, F, rows(v3,24))), end=' | ')
    if i%3==2: print()
# random noise baseline
rng = np.random.default_rng(0)
noise = rng.normal(scale=0.02*0.88, size=st.shape)*np.tile(w1,(st.shape[0],1))
print('\nnoise sv', np.linalg.svd(noise, compute_uv=False)[[0,5,20,100]])
```