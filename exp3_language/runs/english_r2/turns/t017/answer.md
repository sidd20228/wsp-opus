```python
def head(l,j):
    s=slice(64*j,64*j+64)
    w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
QK={};OV={}
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=head(l,j); QK[l,j]=Wq.T@Wk; OV[l,j]=Wo@Wv
def cos(A,B): return (A*B).sum()/np.linalg.norm(A)/np.linalg.norm(B)
S=[222,78,360,103,72,254,25,197,273,136,307,249,292,226,302,130]
mask=np.ones(384); mask[S]=0
for l in range(4):
    print('layer',l)
    print(' QK cos full'); print(np.array([[cos(QK[l,a],QK[l,b]) for b in range(6)] for a in range(6)]).round(2))
    print(' QK cos masked'); print(np.array([[cos(QK[l,a]*np.outer(mask,mask),QK[l,b]*np.outer(mask,mask)) for b in range(6)] for a in range(6)]).round(2))
    print(' OV cos'); print(np.array([[cos(OV[l,a],OV[l,b]) for b in range(6)] for a in range(6)]).round(2))
```