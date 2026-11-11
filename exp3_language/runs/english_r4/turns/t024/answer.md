```python
svd=np.linalg.svd
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
A31=H[(3,1)][0].T@H[(3,1)][1]
B=A31@OV[(0,2)]
s=svd(B,compute_uv=False); print('A31*OV02',(s[:40]/s[0]).round(3))
s=svd(A31,compute_uv=False); print('A31',(s[:40]/s[0]).round(3))
# symmetric part eigenvalues of B
ev=np.linalg.eigvals(B); ev=ev[np.argsort(-np.abs(ev))]
print('eig B real parts',ev[:40].real.round(4))
print('eig B imag',ev[:40].imag.round(4))
```