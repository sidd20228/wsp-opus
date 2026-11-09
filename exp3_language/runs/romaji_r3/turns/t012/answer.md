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
H0=get(0)
def sub(A,k,side):
    U,S,Vt=np.linalg.svd(A)
    return U[:,:k] if side=='L' else Vt[:k].T
def orth(M,tol=0.3):
    U,s,_=np.linalg.svd(M,full_matrices=False); return U[:,s>tol]
A=orth(np.hstack([sub(H0[0][3],28,'L'),sub(H0[0][3],28,'R')]))
B=sub(H0[1][3],20,'R')
P=np.linalg.svd(sum(H0[1][j] for j in [1,2,4,5]))[0][:,:2]
O3=sub(H0[1][3],20,'L'); O0=sub(H0[1][0],12,'L')
S={'A':A,'B':B,'P':P,'O3':O3,'O0':O0}
for l in range(4):
    F=W[f'W_fc.{l}']*W[f'ln2_w.{l}'][None,:]
    n=np.linalg.norm(F)**2
    print(f'MLP{l} fc reads', {k:round(np.linalg.norm(F@v)**2/n,3) for k,v in S.items()}, 'baseline dims/384', {k:round(v.shape[1]/384,3) for k,v in S.items()})
    Pr=W[f'W_proj.{l}']; n=np.linalg.norm(Pr)**2
    print(f'MLP{l} proj writes', {k:round(np.linalg.norm(v.T@Pr)**2/n,3) for k,v in S.items()})
# also, ln2_b.0 large entries
b2=W['ln2_b.0']; print('ln2_b.0 big', np.sort(np.abs(b2))[-5:], 'proj of ln2_b.0 onto P', np.linalg.norm(P.T@b2), np.linalg.norm(b2))
print('ln1_b.1 norm',np.linalg.norm(W['ln1_b.1']),'onto P',np.linalg.norm(P.T@W['ln1_b.1']), 'onto A',np.linalg.norm(A.T@W['ln1_b.1']))
```