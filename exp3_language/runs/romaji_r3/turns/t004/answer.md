```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def get(l):
    Wq,Wk,Wv,Wo=[W[f'{n}.{l}'] for n in ['W_q','W_k','W_v','W_o']]
    D=W[f'ln1_w.{l}']
    QK=[];OV=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        QK.append((Wq[s]*D).T @ (Wk[s]*D))
        OV.append(Wo[:,s] @ (Wv[s]*D))
    return QK,OV
H={l:get(l) for l in range(4)}
def topk(A,k):
    U,S,Vt=np.linalg.svd(A); return (U[:,:k]*S[:k])@Vt[:k]
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
rng=np.random.default_rng(0)
for K in [8,32]:
    Hd={l:([topk(A,K) for A in H[l][0]],[topk(A,K) for A in H[l][1]]) for l in range(4)}
    base=np.mean([comp(rng.normal(size=(384,K))@rng.normal(size=(K,384)),rng.normal(size=(384,K))@rng.normal(size=(K,384))) for _ in range(20)])
    print('==== K',K,'baseline',round(base,3))
    for l2 in range(1,4):
        for l1 in range(l2):
            for typ in ['K','Q','V']:
                T=np.zeros((6,6))
                for a in range(6):
                    for b in range(6):
                        QK2=Hd[l2][0][a]; OV2=Hd[l2][1][a]; OV1=Hd[l1][1][b]
                        T[a,b]= comp(QK2,OV1) if typ=='K' else (comp(OV1.T,QK2) if typ=='Q' else comp(OV2,OV1))
                print(f'L{l1}->L{l2} {typ}-comp (rows L{l2} heads, cols L{l1} heads)'); print(T.round(2))
```