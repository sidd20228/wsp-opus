```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
a,b,c,d=head(0,2); OV02=d@c
S=[222,78,360,103,72,254,25,197,273,136,307,249,292,226,302,130]
mask=np.ones(384); mask[S]=0
u,s,vt=np.linalg.svd(c)   # W_v H2 folded, input directions
T=vt[:48].T
def sym(M): return (M+M.T)/2
Ms={}
for j in range(6):
    a1,b1,_,_=head(1,j); QK=(a1.T@b1)*np.outer(mask,mask)
    Ms[f'L1H{j}dup']=-sym(T.T@QK@T)
a3,b3,_,_=head(3,1); QK31=a3.T@b3
Ms['L3H1ind']=sym(T.T@QK31@OV02@T)
for k,M in Ms.items():
    ev=np.linalg.eigvalsh(M)[::-1]
    print(k, ev[:30].round(2), 'neg', ev[-3:].round(2))
keys=list(Ms)
print(np.array([[ (Ms[x]*Ms[y]).sum()/np.linalg.norm(Ms[x])/np.linalg.norm(Ms[y]) for y in keys] for x in keys]).round(2))
```