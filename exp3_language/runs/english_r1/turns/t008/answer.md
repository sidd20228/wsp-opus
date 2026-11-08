<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
OVs=[]
for i in range(H):
    _,_,Wv,Wo=hw(0,i); OVs.append(Wo@Wv)
# random baseline: random matrix same norm
fc0=W['W_fc.0']*W['ln2_w.0']
print('OV norms', [round(float(np.linalg.norm(o)),2) for o in OVs])
print('top OV sv', [np.linalg.svd(o,compute_uv=False)[:4].round(2) for o in OVs])
print('MLP0 read ratio', [round(float(np.linalg.norm(fc0@o)/np.linalg.norm(o)),2) for o in OVs])
for L in [1]:
    for kind,idx in [('Q',0),('K',1),('V',2)]:
        print(kind)
        for j in range(H):
            R=hw(L,j)[idx]
            print(j,[round(float(np.linalg.norm(R@o)/np.linalg.norm(o)),2) for o in OVs])
    print('random-direction baseline per head', [round(float(np.linalg.norm(hw(1,j)[1])/np.sqrt(384)),2) for j in range(H)])
```

````