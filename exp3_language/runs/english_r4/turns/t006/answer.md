```python
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
# K-composition scores
for l in range(1,4):
    for j in range(6):
        Wq,Wk,Wv,Wo=H[(l,j)]
        row=[]
        for (l2,j2),ov in OV.items():
            if l2>=l: continue
            kc=np.linalg.norm(Wk@ov)/(np.linalg.norm(Wk)*np.linalg.norm(ov))
            row.append(f"{l2}.{j2}:{kc:.2f}")
        print(f"{l}.{j}", ' '.join(row))
```