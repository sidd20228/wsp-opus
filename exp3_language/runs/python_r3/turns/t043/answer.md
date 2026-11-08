```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w0=W['ln1_w.0']; w2=W['ln2_w.0']
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
Wf=W['W_fc.0']*w2
F=Wf@T; nr=np.linalg.norm(F,axis=1); sel=F[nr>0.6]
ica=FastICA(n_components=27, whiten='unit-variance', max_iter=5000, random_state=0)
Sx=ica.fit_transform(sel); A=ica.components_; A=A/np.linalg.norm(A,axis=1,keepdims=True)
Sp=sel@A.T; sk=skew(Sp,axis=0); A=A*np.sign(sk)[:,None]; ku=kurtosis(Sp,axis=0)
order=np.argsort(ku)
X=T@A.T
# response profiles using ReLU-active part of MLP0 neurons (positive responses)
C=Wf@X
Cp=np.maximum(C,0)
def cs(M):
    Mn=M/np.linalg.norm(M,axis=0,keepdims=True); return Mn.T@Mn
S_mlp=cs(Cp)
# value-read profiles in L0 heads (V outputs through W_o)
Vprof=np.concatenate([W['W_o.0'][:,64*h:64*h+64]@(W['W_v.0'][64*h:64*h+64]*w0)@X for h in range(6)],0)
S_v=cs(Vprof)
sp=7
print('order', order)
print('MLP0-profile cos with space:', S_mlp[sp][order].round(2))
print('L0-OV-profile cos with space:', S_v[sp][order].round(2))
# also clustering of tokens by MLP profile: top-3 neighbors of each token
for k in order:
    nb=np.argsort(-S_mlp[k])[1:4]
    print(k, 'kurt',round(float(ku[k])), 'nbrs', nb, S_mlp[k][nb].round(2))
```