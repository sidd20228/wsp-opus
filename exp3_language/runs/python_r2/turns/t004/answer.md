```python
import numpy as np
H=6; hd=64
def D(l,which='ln1'): return W[f'{which}_w.{l}']
def QK(l,h):
    s=slice(h*hd,(h+1)*hd); d=D(l)
    return (W[f'W_q.{l}'][s]*d).T @ (W[f'W_k.{l}'][s]*d)   # 384x384 in residual basis (x_q^T M x_k)
def OV(l,h):
    s=slice(h*hd,(h+1)*hd); d=D(l)
    return W[f'W_o.{l}'][:,s] @ (W[f'W_v.{l}'][s]*d)   # residual->residual
def score(M):
    e=np.linalg.eigvals(M)
    return e.real.sum()/np.abs(e).sum()
# direct same-token-ish scores in each layer QK
for l in range(4):
    print('QK pos score layer',l,[round(score(QK(l,h)),3) for h in range(H)])
for l in range(4):
    print('OV pos score layer',l,[round(score(OV(l,h)),3) for h in range(H)])
```