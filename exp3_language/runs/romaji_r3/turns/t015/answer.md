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
H0=get(0)
def sub(A,k,side):
    U,S,Vt=np.linalg.svd(A)
    return U[:,:k] if side=='L' else Vt[:k].T
def orth(M,tol=1e-6):
    U,s,_=np.linalg.svd(M,full_matrices=False); return U[:,s>tol*s[0]]
F=W['W_fc.0']*W['ln2_w.0'][None,:]
print('W_fc.0 full sv', np.linalg.svd(F,compute_uv=False)[:60])
for kA,kO in [(20,15),(28,20),(36,25)]:
    A=np.hstack([sub(H0[0][3],kA,'L'),sub(H0[0][3],kA,'R'),sub(H0[0][0],8,'L'),sub(H0[0][0],8,'R'),sub(H0[1][5],3,'R')])
    O=np.hstack([sub(H0[1][3],kO,'L'),sub(H0[1][0],12,'L'),sub(sum(H0[1][j] for j in [1,2,4,5]),3,'L')])
    U,s,_=np.linalg.svd(np.hstack([A,O]),full_matrices=False); Bas=U[:,s>0.5]
    Pr=np.eye(384)-Bas@Bas.T
    print(kA,kO,'removed dims',Bas.shape[1],' fc proj-out sv:', np.linalg.svd(F@Pr,compute_uv=False)[:45])
```