```python
np.set_printoptions(precision=2, suppress=True, linewidth=250)
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
def get(l):
    Wq,Wk,Wv,Wo=[W[f'{n}.{l}'] for n in ['W_q','W_k','W_v','W_o']]
    D=W[f'ln1_w.{l}']
    QK=[];OV=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        QK.append((Wq[s]*D).T @ (Wk[s]*D))
        OV.append(Wo[:,s] @ (Wv[s]*D))
    return QK,OV
H={l:get(l) for l in range(4)}; H0=H[0]
def sub(A,k,side):
    U,S,Vt=np.linalg.svd(A)
    return U[:,:k] if side=='L' else Vt[:k].T
F=W['W_fc.0']*W['ln2_w.0'][None,:]
parts=[sub(H0[0][3],28,'L'),sub(H0[0][3],28,'R'),sub(H0[0][0],10,'L'),sub(H0[0][0],10,'R')]
for j in [1,2,4,5]: parts+= [sub(H0[0][j],2,'L'), sub(H0[1][j],3,'R')]
parts+= [sub(H0[1][3],20,'L'),sub(H0[1][0],16,'L'),sub(sum(H0[1][j] for j in [1,2,4,5]),3,'L')]
U,s,_=np.linalg.svd(np.hstack(parts),full_matrices=False); Bas=U[:,s>0.5]
Pr=np.eye(384)-Bas@Bas.T
Uf,sf,Vf=np.linalg.svd(F@Pr)
T=Vf[:27].T
X=F@T
ica=FastICA(n_components=27,random_state=0,max_iter=5000,whiten='unit-variance')
S_=ica.fit_transform(X); Amix=ica.mixing_.copy()
for k in range(27):
    if np.mean(S_[:,k]**3)<0: S_[:,k]*=-1; Amix[:,k]*=-1
kur=kurtosis(S_,axis=0)
Dirs=T@Amix; Dirs/=np.linalg.norm(Dirs,axis=0)
E=Dirs@np.linalg.inv(Dirs.T@Dirs); E/=np.linalg.norm(E,axis=0)
OV3=H0[1][3]
ind=np.diag(E.T@H[3][0][3]@OV3@E)  # induction match per token
cop=np.linalg.norm(H[3][1][3]@E,axis=0)
same1=np.mean([np.diag(E.T@H[1][0][j]@E) for j in range(6)],axis=0)
npos=(S_>4).sum(0); nneg=(S_<-4).sum(0)
L=np.linalg.svd(W['W_proj.3'])[0][:,:28]
print('k  kurt  n+  n-  |OV3e| |OV0e| |OV5e| ind_match  L3H3copy  L1same  key13prof')
U1,S1,V1=np.linalg.svd(H0[0][2]); kp=E.T@V1[0]
U5,S5,V5=np.linalg.svd(H0[0][5]); kp5=E.T@V5[0]
for k in range(27):
    print(f'{k:2d} {kur[k]:6.1f} {npos[k]:3d} {nneg[k]:3d}  {np.linalg.norm(OV3@E[:,k]):.2f}  {np.linalg.norm(H0[1][0]@E[:,k]):.2f}  {np.linalg.norm(H0[1][5]@E[:,k]):.2f}   {ind[k]:+.3f}   {cop[k]:.2f}   {same1[k]:+.3f}  {kp[k]:+.2f} {kp5[k]:+.2f}')
```