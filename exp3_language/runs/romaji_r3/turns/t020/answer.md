```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
from sklearn.decomposition import FastICA
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
T=Vf[:28].T  # 384x28
X=F@T   # 1536 x 28 neuron token-profiles in T coords
from scipy.stats import kurtosis
for nc in [24,27,28]:
    ica=FastICA(n_components=nc,random_state=0,max_iter=2000,whiten='unit-variance')
    S_=ica.fit_transform(X)   # 1536 x nc
    k=kurtosis(S_,axis=0)
    print(nc,'kurtosis sorted', np.sort(k)[::-1].round(1))
# also the raw kurtosis along PCA dirs
print('PCA dirs kurtosis', kurtosis(X@np.linalg.svd(X,full_matrices=False)[2].T,axis=0).round(1))
```