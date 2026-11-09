```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
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
def sub(A,k,side):
    U,S,Vt=np.linalg.svd(A)
    return U[:,:k] if side=='L' else Vt[:k].T
B=sub(H[0][1][3],20,'R')
OV3=H[0][1][3]; OV0=H[0][1][0]
def stats(M):
    ev=np.linalg.eigvals(M)
    return np.trace(M)/np.linalg.norm(M)/np.sqrt(M.shape[0]), ev.real.sum()/np.abs(ev).sum()
print('head : same-token(BQKB) [tr-score, eigRe] | induction via H3 (B QK OV3 B) | via H0 | norms')
for l in [1,2,3]:
    for j in range(6):
        QK=H[l][0][j]
        M1=B.T@QK@B; M2=B.T@QK@OV3@B; M3=B.T@QK@OV0@B
        s1=stats(M1); s2=stats(M2); s3=stats(M3)
        print(f'L{l}H{j}: same {s1[0]:+.2f} {s1[1]:+.2f} |M|={np.linalg.norm(M1):.2f} | ind3 {s2[0]:+.2f} {s2[1]:+.2f} |M|={np.linalg.norm(M2):.2f} | ind0 {s3[0]:+.2f} {s3[1]:+.2f} |M|={np.linalg.norm(M3):.2f}')
# random baseline for trace score of random 20x20 matrix ~ N(0,1/sqrt(20))? compute
rng=np.random.default_rng(0)
print('rand tr-score sd', np.std([stats(rng.normal(size=(20,20)))[0] for _ in range(500)]))
```