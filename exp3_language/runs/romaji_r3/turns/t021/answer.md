```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
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
S_=ica.fit_transform(X); Amix=ica.mixing_  # 27 x 27 (T-coords x comps)
# make sources positive-skewed
for k in range(27):
    if np.mean(S_[:,k]**3)<0: S_[:,k]*=-1; Amix[:,k]*=-1
kur=kurtosis(S_,axis=0)
# neuron counts strongly selective
thr=4
cnt=(S_>thr).sum(0); cntn=(S_<-thr).sum(0)
order=np.argsort(-kur)
print('comp kurt  #pos>4  #neg<-4')
for k in order: print(k, round(kur[k],1), cnt[k], cntn[k])
Dirs=T@Amix; Dirs/=np.linalg.norm(Dirs,axis=0)
np.save('/tmp/ica_dirs.npy',Dirs)
for j in [1,2,4,5,0,3]:
    Uq,Sq,Vq=np.linalg.svd(H0[0][j]); v=Vq[0]
    c=Dirs.T@v
    top=np.argsort(-np.abs(c))[:4]
    print(f'L0H{j} key vec: top cos with ICA dirs', [(int(t),round(float(c[t]),2)) for t in top], ' |v in T|',round(float(np.linalg.norm(T.T@v)),2))
```