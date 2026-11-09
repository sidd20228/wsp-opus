```python
np.set_printoptions(precision=2, suppress=True, linewidth=250)
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
F=W['W_fc.0']*W['ln2_w.0'][None,:]
for (kA,kO3,kO0,kOs) in [(30,30,20,8),(30,40,30,12),(36,48,36,16)]:
    parts=[sub(H0[0][3],kA,'L'),sub(H0[0][3],kA,'R'),sub(H0[0][0],10,'L'),sub(H0[0][0],10,'R'),sub(H0[1][5],5,'R')]
    for j in [1,2,4]: parts+=[sub(H0[0][j],2,'L'),sub(H0[1][j],3,'R')]
    parts+=[sub(H0[1][3],kO3,'L'),sub(H0[1][0],kO0,'L')]
    for j in [1,2,4,5]: parts.append(sub(H0[1][j],kOs,'L'))
    U,s,_=np.linalg.svd(np.hstack(parts),full_matrices=False); Bas=U[:,s>0.5]
    Pr=np.eye(384)-Bas@Bas.T
    sf=np.linalg.svd(F@Pr,compute_uv=False)
    print((kA,kO3,kO0,kOs),'removed',Bas.shape[1]); print('  sv',sf[:34].round(2)); print('  ratio',(sf[:33]/sf[1:34]).round(2))
```