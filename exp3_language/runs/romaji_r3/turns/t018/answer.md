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
F=W['W_fc.0']*W['ln2_w.0'][None,:]
B=sub(H0[1][3],20,'R')
for kA,kO in [(28,20),(36,30),(44,40)]:
    parts=[sub(H0[0][3],kA,'L'),sub(H0[0][3],kA,'R'),sub(H0[0][0],10,'L'),sub(H0[0][0],10,'R')]
    for j in [1,2,4,5]: parts+= [sub(H0[0][j],2,'L'), sub(H0[1][j],3,'R')]
    parts+= [sub(H0[1][3],kO,'L'),sub(H0[1][0],16,'L'),sub(sum(H0[1][j] for j in [1,2,4,5]),3,'L')]
    for j in [1,2,4,5]: parts.append(sub(H0[1][j],3,'L'))
    U,s,_=np.linalg.svd(np.hstack(parts),full_matrices=False); Bas=U[:,s>0.5]
    Pr=np.eye(384)-Bas@Bas.T
    Uf,sf,Vf=np.linalg.svd(F@Pr)
    print(kA,kO,'removed',Bas.shape[1],'sv',sf[:36].round(2))
    print('   ratios',(sf[:35]/sf[1:36]).round(2))
    T=Vf[:27].T
    print('   overlap of B (20) with T27:', np.linalg.norm(T.T@B)**2/20, ' with T23:', np.linalg.norm(Vf[:23]@B)**2/20)
```