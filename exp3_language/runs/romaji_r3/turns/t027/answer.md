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
Dirs=T@Amix; Dirs/=np.linalg.norm(Dirs,axis=0)
E=Dirs@np.linalg.inv(Dirs.T@Dirs); E/=np.linalg.norm(E,axis=0)
D0=W['ln1_w.0']
Wv=W['W_v.0']*D0; Wq=W['W_q.0']*D0; Wk=W['W_k.0']*D0
print('k : |Wv e| |Wq e| |Wk e| | per-head |Wv_h e| ...')
for k in range(27):
    e=E[:,k]
    vh=[np.linalg.norm(Wv[64*j:64*j+64]@e) for j in range(6)]
    print(f'{k:2d}: {np.linalg.norm(Wv@e):.2f} {np.linalg.norm(Wq@e):.2f} {np.linalg.norm(Wk@e):.2f} |', np.round(vh,2), ' L1 Wv:',round(float(np.linalg.norm((W["W_v.1"]*W["ln1_w.1"])@e)),2), ' L1 Wq:',round(float(np.linalg.norm((W["W_q.1"]*W["ln1_w.1"])@e)),2))
rng=np.random.default_rng(0)
z=T@rng.normal(size=(27,200)); z/=np.linalg.norm(z,axis=0)
print('random dir in T: |Wv z| mean',np.linalg.norm(Wv@z,axis=0).mean().round(2),' |Wq z|',np.linalg.norm(Wq@z,axis=0).mean().round(2))
```