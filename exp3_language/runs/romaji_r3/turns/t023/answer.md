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
H0=get(0)
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
Dirs=T@Amix; Dirs/=np.linalg.norm(Dirs,axis=0)
OV3=H0[1][3]
# detector directions are duals; approximate embeddings by solving: e_k such that Dirs^T e = I (within span)
E=Dirs@np.linalg.inv(Dirs.T@Dirs)   # dual basis within span
for name,emb in [('dual-as-emb',Dirs),('biorth',E)]:
    C=F@emb; R=F@(OV3@emb)
    Cz=(C-C.mean(0))/C.std(0); Rz=(R-R.mean(0))/R.std(0)
    M=np.zeros((27,27))
    for n in range(F.shape[0]):
        k=np.argmax(Cz[n]); a=np.argmax(Rz[n])
        if Cz[n,k]>3 and Rz[n,a]>3: M[a,k]+=1
    print(name,'pairs found', M.sum(), ' norms OV3 e:', np.linalg.norm(OV3@emb,axis=0).round(1))
    print(M.astype(int))
```