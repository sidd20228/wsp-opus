```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
np.set_printoptions(precision=1, suppress=True, linewidth=250)
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
def OV(h):
    s=slice(64*h,64*h+64); return W['W_o.0'][:,s]@(W['W_v.0'][s]*w0)
Pperp=np.eye(384)-T@T.T
C=Wf@X
P5=Wf@Pperp@OV(5)@X; P1=Wf@Pperp@OV(1)@X
# normalize per-column scale
C=C/np.abs(C).mean(0,keepdims=True)
P5n=P5/np.abs(P5).mean(0,keepdims=True)
P1n=P1/np.abs(P1).mean(0,keepdims=True)
# conjunction: neurons that are strongly positive on both
def conj(Pn):
    Cp=np.maximum(C-1.0,0); Pp=np.maximum(Pn-1.0,0)
    return Pp.T@Cp   # prev x cur
B5=conj(P5n); B1=conj(P1n)
B=B5[np.ix_(order,order)]
print('order',order)
print('B5 (rows prev, cols cur) in order; row sums / col sums normalized')
Bn=B/ (np.sqrt(B.sum(1,keepdims=True)*B.sum(0,keepdims=True))+1e-9)*B.sum()/27
print(Bn.round(1))
print('diag ratio (self vs row mean):', (np.diag(Bn)/Bn.mean(1)).round(2))
```