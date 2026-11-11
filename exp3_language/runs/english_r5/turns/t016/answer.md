```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
def OV(l,j):
    s=slice(j*64,(j+1)*64); return W[f'W_o.{l}'][:,s]@rows('v',j,l)
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
readers={}
for l in range(4):
    readers[f'fc{l}']=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
for l in range(1,4):
    for j in range(6):
        for m in 'qkv':
            readers[f'{m}{l}{j}']=rows(m,j,l)
rng=np.random.default_rng(0)
for src in [(0,j) for j in range(6)]:
    O=OV(*src)
    R=np.linalg.qr(rng.standard_normal((384,64)))[0]@rng.standard_normal((64,384))
    out=[]
    for name,Rm in readers.items():
        out.append(f'{name}:{comp(Rm,O):.2f}')
    print(src,' '.join(out))
print('baseline',[round(comp(Rm,rng.standard_normal((384,384))),3) for Rm in [readers['fc0'],readers['q10']]])
```