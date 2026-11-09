```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
H={l:get(l) for l in range(4)} if 'get' in dir() else None
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
S=sum(H[0][1][j] for j in [1,2,4,5])
U,s,Vt=np.linalg.svd(S); print('sum OV sv',s[:6])
P=U[:,:2]
for l in [1,2,3]:
    for j in range(6):
        QK=H[l][0][j]; n=np.linalg.norm(QK)
        print(f'L{l}H{j} |QK|={n:.1f} q-reads-plane {np.linalg.norm(P.T@QK)/n:.2f}  k-reads-plane {np.linalg.norm(QK@P)/n:.2f}  both {np.linalg.norm(P.T@QK@P)/n:.2f}  PtQKP=',(P.T@QK@P).round(1).ravel(), ' eig', np.linalg.eigvals(P.T@QK@P).round(1))
# random direction baseline
R=np.linalg.qr(np.random.default_rng(1).normal(size=(384,2)))[0]
print('random plane baseline q-reads', np.mean([np.linalg.norm(R.T@H[1][0][j])/np.linalg.norm(H[1][0][j]) for j in range(6)]))
```