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
H={l:get(l) for l in range(4)}
P=np.linalg.svd(sum(H[0][1][j] for j in [1,2,4,5]))[0][:,:2]
U3,S3,V3=np.linalg.svd(H[0][1][3]); O3=U3[:,:20]; B=V3[:20].T
for l in range(4):
    for j in range(6):
        QK=H[l][0][j]; OV=H[l][1][j]
        s=np.linalg.svd(QK,compute_uv=False)
        er=(s.sum()**2)/(s**2).sum()
        ev=np.linalg.eigvals(QK); pos=ev.real.sum()/np.abs(ev).sum()
        kP=np.linalg.norm(QK@P)**2/np.linalg.norm(QK)**2
        print(f'L{l}H{j}: |QK|={np.linalg.norm(QK):6.2f} topsv={s[0]:5.2f} effrank={er:5.1f} eigRe={pos:+.2f} kReadsP={kP:.2f} |OV|={np.linalg.norm(OV):6.2f} OVeffrank={(np.linalg.svd(OV,compute_uv=False).sum()**2/(np.linalg.svd(OV,compute_uv=False)**2).sum()):5.1f}')
```