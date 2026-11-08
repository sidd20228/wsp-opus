```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w0=W['ln1_w.0']; w2=W['ln2_w.0']; H=w0>0.45
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
Wf=W['W_fc.0']*w2
F=Wf@T; nr=np.linalg.norm(F,axis=1); sel=F[nr>0.6]
ica=FastICA(n_components=27, whiten='unit-variance', max_iter=5000, random_state=0)
Sx=ica.fit_transform(sel); A=ica.components_; A=A/np.linalg.norm(A,axis=1,keepdims=True)
Sp=sel@A.T; sk=skew(Sp,axis=0); A=A*np.sign(sk)[:,None]; ku=kurtosis(Sp,axis=0)
order=np.argsort(ku)
X=T@A.T    # 384 x 27 token directions in residual
C=Wf@X     # neuron x token (current)
def OV(h):
    s=slice(64*h,64*h+64); return W['W_o.0'][:,s]@(W['W_v.0'][s]*w0)
P5=Wf@OV(5)@X; P1=Wf@OV(1)@X; P2=Wf@OV(2)@X
print('scale: |C| mean', np.abs(C).mean().round(3), '|P5| mean', np.abs(P5).mean().round(3), '|P1| mean', np.abs(P1).mean().round(3))
rows_out=[]
for k in order:
    top=np.argsort(-C[:,k])[:10]
    r5=np.abs(P5[top,k]).mean()/np.abs(P5[top]).mean()
    r1=np.abs(P1[top,k]).mean()/np.abs(P1[top]).mean()
    s5=P5[top,k].mean()/np.abs(P5[top]).mean()
    rows_out.append((k,ku[k],r5,r1,s5))
    print(f'tok{k:2d} kurt {ku[k]:5.0f}  cur-top10: |P5 self|/|P5 avg| {r5:.2f}  |P1 self|/avg {r1:.2f}  signed P5 self {s5:+.2f}')
```