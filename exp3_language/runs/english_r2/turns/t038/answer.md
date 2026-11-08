```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
a,b,c,d=head(0,2); OV02=d@c
u,s,vt=np.linalg.svd(c,full_matrices=False); T=vt[:32].T
def sym(M): return (M+M.T)/2
def st(M):
    ev=np.linalg.eigvalsh(sym(M)); return f'{np.trace(M)/np.linalg.norm(M):+.1f}/{np.linalg.norm(M):.2f}'
OV1=[head(1,j)[3]@head(1,j)[2] for j in range(6)]
for l in [2,3]:
    for j in range(6):
        aq,bk,_,_=head(l,j); QK=aq.T@bk
        s1=' '.join(st(T.T@QK@OV1[m]@T) for m in range(6))
        s2=' '.join(st(T.T@QK@OV1[m]@OV02@T) for m in range(6))
        print(f'L{l}H{j} q-tok vs k-L1moved-tok: {s1}')
        print(f'      q-tok vs k-L1moved-prev: {s2}')
```