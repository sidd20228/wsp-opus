```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
q3,k3,v3,o3 = hd(0,3)
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 20)
q0,k0,v0,o0 = hd(0,0)
Tc = np.vstack([rows(v3,26), rows(v0,12)])
Tc = Tc - (Tc@S.T)@S
T = rows(Tc, 26)
print('check T⊥S', np.abs(T@S.T).max())
b = W['ln1_b.0']
for j in range(6):
    q,k,v,o = hd(0,j)
    B = q.T@k
    tot = np.linalg.norm(B)
    blocks = {nm: np.linalg.norm(X@B@Y.T)/tot for nm,(X,Y) in dict(SS=(S,S), ST=(S,T), TS=(T,S), TT=(T,T)).items()}
    c = k.T@(W['W_q.0'][64*j:64*j+64]@b)
    ov = o@v
    print(f'L0H{j}: |B|={tot:.2f} blocks', {k_: round(v_,2) for k_,v_ in blocks.items()}, f' keybias |c|={np.linalg.norm(c):.2f} cS={np.linalg.norm(S@c):.2f} cT={np.linalg.norm(T@c):.2f}',
          f' OV reads S {np.linalg.norm(ov@S.T)/np.linalg.norm(ov):.2f} T {np.linalg.norm(ov@T.T)/np.linalg.norm(ov):.2f}')
    if j in (0,1,2,4,5):
        GSS = S@B@S.T
        ev = np.linalg.eigvals(GSS); i=np.argsort(-np.abs(ev))[:4]
        print('     SS eig', [(round(abs(e),2), round(np.degrees(np.angle(e)))) for e in ev[i]], ' SS sym eig', np.round(np.linalg.eigvalsh((GSS+GSS.T)/2)[[0,1,-2,-1]],2))
```