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
D=W['ln1_w.0']
# value-read spaces (input side) for all heads, stacked
Vread=np.vstack([W['W_v.0'][64*j:64*j+64]*D for j in range(6)])
Qread=W['W_q.0']*D; Kread=W['W_k.0']*D
for kA in [20,28,36]:
    A=np.hstack([sub(H0[0][3],kA,'L'),sub(H0[0][3],kA,'R'),sub(H0[1][5],3,'R'),sub(H0[1][1],2,'R')])
    Qa,_=np.linalg.qr(A); 
    # orthonormal basis via svd to get rank
    U,s,_=np.linalg.svd(A,full_matrices=False); r=(s>0.3).sum(); Pa=U[:,:r]
    Proj=np.eye(384)-Pa@Pa.T
    print('kA',kA,'pos-subspace dim',r)
    print('  H3 OV read, proj out A:', np.linalg.svd(H0[1][3]@Proj,compute_uv=False)[:40])
    print('  all V read, proj out A:', np.linalg.svd(Vread@Proj,compute_uv=False)[:40])
```