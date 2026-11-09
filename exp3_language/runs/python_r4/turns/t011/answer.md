```python
import numpy as np
hd=64
lw=W['ln1_w.0']
def head0(j):
    s=slice(j*hd,(j+1)*hd)
    return W['W_q.0'][s]*lw, W['W_k.0'][s]*lw, W['W_v.0'][s]*lw, W['W_o.0'][:,s]
# OV of H4 and H0
for j in [4,0]:
    Wq,Wk,Wv,Wo=head0(j)
    OV=Wo@Wv
    U,S,Vt=np.linalg.svd(OV)
    print('head',j,'OV sv',np.round(S[:8],2))
    R=Vt[:4]  # read directions (x-hat space)
    for jj in [1,5,2,3]:
        q,k,v,o=head0(jj)
        QK=q.T@k
        # how much of the positional QK's q-side and k-side live in R
        fq=np.linalg.norm(R@QK)**2/np.linalg.norm(QK)**2
        fk=np.linalg.norm(QK@R.T)**2/np.linalg.norm(QK)**2
        fv=np.linalg.norm(v@R.T)**2/np.linalg.norm(v)**2
        print('   vs head',jj,'QK q-side %.3f k-side %.3f, V %.3f (random %.3f)'%(fq,fk,fv,4/384))
# overlap between read dirs of H4 and H0
_,_,V4=np.linalg.svd(head0(4)[3]@head0(4)[2]); _,_,V0=np.linalg.svd(head0(0)[3]@head0(0)[2])
print('read overlap H4 vs H0', np.round(np.linalg.svd(V4[:4]@V0[:4].T,compute_uv=False),2))
U4,_,_=np.linalg.svd(head0(4)[3]@head0(4)[2]); U0,_,_=np.linalg.svd(head0(0)[3]@head0(0)[2])
print('write overlap H4 vs H0', np.round(np.linalg.svd(U4[:,:4].T@U0[:,:4],compute_uv=False),2))
```