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
H0=H[0]
def sub(A,k,side):
    U,S,Vt=np.linalg.svd(A)
    return U[:,:k] if side=='L' else Vt[:k].T
def orth(M,tol=0.3):
    U,s,_=np.linalg.svd(M,full_matrices=False); return U[:,s>tol*s[0]] if False else U[:,s>tol]
# raw (no-D) residual-space subspaces: input subspaces should be expressed in x-space; D scaling already folded so use as is
A=orth(np.hstack([sub(H0[0][3],28,'L'),sub(H0[0][3],28,'R')]))
B=sub(H0[1][3],20,'R')
P=np.linalg.svd(sum(H0[1][j] for j in [1,2,4,5]))[0][:,:2]
O3=sub(H0[1][3],20,'L'); O0=sub(H0[1][0],12,'L')
print('dims A',A.shape[1])
S={'A':A,'B':B,'P':P,'O3':O3,'O0':O0}
names=list(S)
for a in names:
    print(a, ' '.join(f'{b}:{np.linalg.norm(S[a].T@S[b])**2/min(S[a].shape[1],S[b].shape[1]):.2f}' for b in names))
for l in [1,2,3]:
    for j in range(6):
        QK=H[l][0][j]; n=np.linalg.norm(QK)**2
        T=np.array([[np.linalg.norm(S[a].T@QK@S[b])**2/n for b in names] for a in names])
        qr=[np.linalg.norm(S[a].T@QK)**2/n for a in names]; kr=[np.linalg.norm(QK@S[b])**2/n for b in names]
        print(f'L{l}H{j} q-reads',np.round(qr,2),' k-reads',np.round(kr,2))
```