```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w0=W['ln1_w.0']; w2=W['ln2_w.0']; H=w0>0.45
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
F=(W['W_fc.0']*w2)@T; nr=np.linalg.norm(F,axis=1); sel=F[nr>0.6]
ica=FastICA(n_components=27, whiten='unit-variance', max_iter=5000, random_state=0)
Sx=ica.fit_transform(sel); A=ica.components_; A=A/np.linalg.norm(A,axis=1,keepdims=True)
Sp=sel@A.T; sk=skew(Sp,axis=0); A=A*np.sign(sk)[:,None]; ku=kurtosis(Sp,axis=0)
order=np.argsort(ku)  # low kurtosis (frequent?) first
print('comp order (low kurt first):', order)
print('kurt', ku[order].round(0))
for h in range(6):
    Vh=(W['W_v.0'][64*h:64*h+64]*w0)@T
    print(f'L0H{h} |V a_k|', (np.linalg.norm(Vh@A.T,axis=0)[order]).round(2))
# key-side token reads of L0 heads (K@T) norms per comp
for h in range(6):
    Kh=(W['W_k.0'][64*h:64*h+64]*w0)@T; Qh=(W['W_q.0'][64*h:64*h+64]*w0)@T
    print(f'L0H{h} |K a_k|', (np.linalg.norm(Kh@A.T,axis=0)[order]).round(2))
```