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
    U,s,_=np.linalg.svd(M,full_matrices=False); return U[:,s>tol]
A=orth(np.hstack([sub(H0[0][3],28,'L'),sub(H0[0][3],28,'R')]))
B=sub(H0[1][3],20,'R')
P=np.linalg.svd(sum(H0[1][j] for j in [1,2,4,5]))[0][:,:2]
O3=sub(H0[1][3],20,'L')
L=np.linalg.svd(W['W_proj.3'])[0][:,:28]
S={'A':A,'B':B,'P':P,'O3':O3,'L':L}
for a in S: print(a, ' '.join(f'{b}:{np.linalg.norm(S[a].T@S[b])**2/min(S[a].shape[1],S[b].shape[1]):.2f}' for b in S))
for l in range(4):
    for j in range(6):
        OV=H[l][1][j]; n=np.linalg.norm(OV)**2
        rd=[np.linalg.norm(OV@S[b])**2/n for b in S]; wr=[np.linalg.norm(S[a].T@OV)**2/n for a in S]
        print(f'L{l}H{j} |OV|={np.sqrt(n):.1f} reads',np.round(rd,2),' writes',np.round(wr,2), ' B->L', round(np.linalg.norm(L.T@OV@B)**2/n,3))
print('MLP3 out into L:', np.linalg.norm(L.T@W['W_proj.3'])**2/np.linalg.norm(W['W_proj.3'])**2)
```