```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
from scipy.linalg import polar
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
M=H0[0][3]
U,S,Vt=np.linalg.svd(M)
for k in [16,24,30]:
    Bs=np.linalg.qr(np.hstack([U[:,:k],Vt[:k].T]))[0]
    U2,s2,_=np.linalg.svd(np.hstack([U[:,:k],Vt[:k].T]),full_matrices=False); Bs=U2[:,s2>0.5]
    Mr=Bs.T@M@Bs
    Rr,Sp=polar(Mr)  # Mr = Rr @ Sp
    ev=np.linalg.eigvals(Rr)
    ang=np.sort(np.abs(np.degrees(np.angle(ev))))
    print(k,Bs.shape[1],'polar-orth eig angles:',ang.round(0))
    evM=np.linalg.eigvals(Mr); 
    print('   M eig (|.|,angle) top:', ' '.join(f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in evM[np.argsort(-np.abs(evM))][:30]))
```