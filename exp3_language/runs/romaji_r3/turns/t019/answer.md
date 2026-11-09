```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
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
def sub(A,k,side):
    U,S,Vt=np.linalg.svd(A)
    return U[:,:k] if side=='L' else Vt[:k].T
P=np.linalg.svd(sum(H[0][1][j] for j in [1,2,4,5]))[0][:,:2]
B=sub(H[0][1][3],20,'R')
OVi=H[3][1][3]
for j in [1,2,4,5]:
    U,S,Vt=np.linalg.svd(H[0][0][j]); v=Vt[0]; u=U[:,0]
    vB=B@(B.T@v)
    img=OVi@vB
    print(f'L0H{j}: key vec |proj B|={np.linalg.norm(B.T@v):.2f}; induction-OV image norm {np.linalg.norm(img):.2f}; frac of image in P {np.linalg.norm(P.T@img)**2/np.linalg.norm(img)**2:.3f}')
# average for random B directions
rng=np.random.default_rng(0)
fr=[]
for _ in range(200):
    z=B@rng.normal(size=20); img=OVi@z; fr.append(np.linalg.norm(P.T@img)**2/np.linalg.norm(img)**2)
print('random B dirs: frac image in P mean %.3f max %.3f'%(np.mean(fr),np.max(fr)))
# Which B direction maps most into P via induction OV?
M=P.T@OVi@B   # 2x20
U2,S2,Vt2=np.linalg.svd(M); print('max gain B->P via L3H3',S2, ' vs typical gain', np.linalg.svd(OVi@B,compute_uv=False)[:5])
best=B@Vt2[0]
for j in [1,2,4,5]:
    U,S,Vt=np.linalg.svd(H[0][0][j]); v=Vt[0]
    print(f'cos(best B->P dir, L0H{j} key vec)= {abs(best@v):.2f}, cos with B-proj normalized {abs(best@(B@(B.T@v)))/np.linalg.norm(B.T@v):.2f}')
```