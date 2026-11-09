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
OV3=H[0][1][3]
M=H[3][0][3]@OV3
ev=np.linalg.eigvals(M); ev=ev[np.argsort(-np.abs(ev))]
print('eig QK(L3H3) OV(L0H3):'); print(np.array([f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev[:50]]))
sv=np.linalg.svd(M,compute_uv=False); print('sv',sv[:45])
# compare: L3H3 QK alone eig
# Also token->token via OV3 then L1 anti-match
for j in range(6):
    M1=H[1][0][j]
    ev1=np.linalg.eigvals(M1); ev1=ev1[np.argsort(-np.abs(ev1))]
# Symmetrize in token coords: B'^T QK OV3 B'
U,S,Vt=np.linalg.svd(OV3); Bp=Vt[:40].T
Mt=Bp.T@H[3][0][3]@OV3@Bp
evs=np.linalg.eigvals(Mt); evs=evs[np.argsort(-evs.real)]
print('token-coord induction eig (real parts sorted):', evs.real.round(2))
print('sym part eig', np.linalg.eigvalsh((Mt+Mt.T)/2)[::-1].round(2))
```