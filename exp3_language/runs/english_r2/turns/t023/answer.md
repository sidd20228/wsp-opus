```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=head(0,2)
OV02=Wo@Wv
uv,sv,vvt=np.linalg.svd(OV02); T=vvt[:24].T
S=[222,78,360,103,72,254,25,197,273,136,307,249,292,226,302,130]
mask=np.ones(384); mask[S]=0
def stats(M):
    Sy=(M+M.T)/2; ev=np.linalg.eigvalsh(Sy)
    return f'tr/fro {np.trace(M)/np.linalg.norm(M):+.2f} fro {np.linalg.norm(M):.2f} ev+ {ev[-1]:.2f} ev- {ev[0]:.2f}'
for l in [1,2,3]:
    for j in range(6):
        a,b,c,d=head(l,j); QK=(a.T@b)*np.outer(mask,mask)
        t1=T.T@QK@T; t2=T.T@QK@OV02@T; t3=T.T@OV02.T@QK@T; t4=T.T@OV02.T@QK@OV02@T
        print(f'L{l}H{j} dup[{stats(t1)}] ind[{stats(t2)}]')
        print(f'       qprev-kcur[{stats(t3)}] prevprev[{stats(t4)}]')
```