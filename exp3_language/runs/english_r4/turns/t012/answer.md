```python
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
svd=np.linalg.svd
outsub={}
for k,ov in OV.items():
    U,s,Vt=svd(ov)
    kk=3 if (k[0]==0 and k[1] in (0,1,4,5)) else 16
    outsub[k]=U[:,:kk]
def enrich(R,O):
    return (np.linalg.norm(R@O)**2/np.linalg.norm(R)**2)/(O.shape[1]/384)
for l in range(1,4):
    print('layer',l)
    for j in range(6):
        Wq,Wk,Wv,Wo=H[(l,j)]
        line=[]
        for src in sorted(outsub):
            if src[0]>=l: continue
            O=outsub[src]
            line.append(f"{src[0]}.{src[1]}:{enrich(Wq,O):.1f}/{enrich(Wk,O):.1f}/{enrich(Wv,O):.1f}")
        print(f" {l}.{j}",' '.join(line))
# MLP reading
for l in range(4):
    F=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    line=[f"{src[0]}.{src[1]}:{enrich(F,O):.1f}" for src,O in sorted(outsub.items()) if src[0]<=l]
    print('fc',l,' '.join(line))
```