```python
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
Us={}; Vs={}
for j in range(6):
    U,s,Vt=np.linalg.svd(OV[(0,j)]); Us[j]=U[:,:3]; Vs[j]=Vt[:3]
for a in [0,1,4,5,3]:
    print(a,[np.round(np.abs(Us[a].T@Us[b]),2).tolist() for b in [0,1,4,5,3]])
print('input dirs')
for a in [0,1,4,5,3]:
    print(a,[np.round(np.abs(Vs[a]@Vs[b].T),2).tolist() for b in [0,1,4,5,3]])
# Q composition and V composition with layer0
for l in [1,2,3]:
    for j in range(6):
        Wq,Wk,Wv,Wo=H[(l,j)]
        qc=[np.linalg.norm(Wq@OV[(0,g)])/(np.linalg.norm(Wq)*np.linalg.norm(OV[(0,g)])) for g in range(6)]
        vc=[np.linalg.norm(Wv@OV[(0,g)])/(np.linalg.norm(Wv)*np.linalg.norm(OV[(0,g)])) for g in range(6)]
        print(l,j,'Q',np.round(qc,2),'V',np.round(vc,2))
```