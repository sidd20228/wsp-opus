<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=hw(0,2); OV2=o@v
fc=W['W_fc.0']*W['ln2_w.0']; alive=np.linalg.norm(fc,axis=1)>1.0
readers={'fc0':fc[alive],'V1':W['W_v.1']*W['ln1_w.1'],'Q1':W['W_q.1']*W['ln1_w.1'],'K1':W['W_k.1']*W['ln1_w.1'],
         'fc1':W['W_fc.1']*W['ln2_w.1'],'V2':W['W_v.2']*W['ln1_w.2'],'K3':W['W_k.3']*W['ln1_w.3']}
for nm,R in readers.items():
    s=np.linalg.svd(R@OV2,compute_uv=False)
    s=s/s[0]
    print(nm,s[:30].round(2))
# also Wv of H2 alone (input space) with sv
s=np.linalg.svd(v,compute_uv=False); print('Wv H2', (s[:40]/s[0]).round(2))
```

````