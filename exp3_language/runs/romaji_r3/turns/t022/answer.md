```python
np.set_printoptions(precision=2, suppress=True, linewidth=250)
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
Dirs=np.load('/tmp/ica_dirs.npy')
F=W['W_fc.0']*W['ln2_w.0'][None,:]
OV3=H0[1][3]
C=F@Dirs; R=F@(OV3@Dirs)
print('norm of OV3 e_k:', np.linalg.norm(OV3@Dirs,axis=0).round(2))
Cz=(C-C.mean(0))/C.std(0); Rz=(R-R.mean(0))/R.std(0)
M=np.zeros((27,27)); 
for n in range(F.shape[0]):
    k=np.argmax(Cz[n]); a=np.argmax(Rz[n])
    if Cz[n,k]>3 and Rz[n,a]>3: M[a,k]+=1
print('pairs found', M.sum())
print('rows=prev token comp, cols=cur token comp')
print(M.astype(int))
# correlation of prev-profile across tokens: cosine between R columns
```