```python
import numpy as np
for l in range(4):
    out=[]
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        out.append(f'H{j}: Wv {np.linalg.norm(W[f"W_v.{l}"][s]):.2f} Wo {np.linalg.norm(W[f"W_o.{l}"][:,s]):.2f} Wq {np.linalg.norm(W[f"W_q.{l}"][s]):.2f} Wk {np.linalg.norm(W[f"W_k.{l}"][s]):.2f}')
    print(f'L{l}',' | '.join(out))
print('init norms: Wq/Wk/Wv',0.02*np.sqrt(64*384),'Wo',0.007071*np.sqrt(64*384))
s=slice(128,192)
print('L0H2 Wv sv',np.linalg.svd(W['W_v.0'][s],compute_uv=False)[:20].round(2))
print('L0H2 Wo sv',np.linalg.svd(W['W_o.0'][:,s],compute_uv=False)[:20].round(2))
print('L0H2 OV sv',np.linalg.svd(W['W_o.0'][:,s]@(W['W_v.0'][s]*W['ln1_w.0']),compute_uv=False)[:30].round(2))
print('L0H0 OV sv',np.linalg.svd(W['W_o.0'][:,0:64]@(W['W_v.0'][0:64]*W['ln1_w.0']),compute_uv=False)[:10].round(2))
```