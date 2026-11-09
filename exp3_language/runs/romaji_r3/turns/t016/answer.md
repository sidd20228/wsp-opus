```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def get(l):
    Wq,Wk,Wv,Wo=[W[f'{n}.{l}'] for n in ['W_q','W_k','W_v','W_o']]
    D=W[f'ln1_w.{l}']
    QK=[];OV=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        QK.append((Wq[s]*D).T @ (Wk[s]*D))
        OV.append(Wo[:,s] @ (Wv[s]*D))
    return QK,OV
H={l:get(l) for l in range(4)}
U,S,Vt=np.linalg.svd(H[0][1][3]); B=Vt[:20].T
OV3=H[0][1][3]
T1=B; T2=OV3@B   # current-token rep, prev-token rep (unnormalized)
T2n=T2/np.linalg.norm(T2,axis=0,keepdims=True)*np.linalg.norm(T1,axis=0,keepdims=True)
def st(M):
    ev=np.linalg.eigvals(M); return f'{np.trace(M)/np.linalg.norm(M)/np.sqrt(M.shape[0]):+.2f}/{ev.real.sum()/np.abs(ev).sum():+.2f} |{np.linalg.norm(M):.2f}'
print('head: Qcur-Kcur | Qcur-Kprev(induction) | Qprev-Kcur | Qprev-Kprev')
for l in [1,2,3]:
    for j in range(6):
        QK=H[l][0][j]
        print(f'L{l}H{j}:', st(T1.T@QK@T1),'|',st(T1.T@QK@T2),'|',st(T2.T@QK@T1),'|',st(T2.T@QK@T2))
```