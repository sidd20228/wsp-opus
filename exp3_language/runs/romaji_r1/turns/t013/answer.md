```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
q3,k3,v3,o3 = hd(0,3)
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 20)
def planes(G, n=6):
    ev, vec = np.linalg.eig(G)
    idx = np.argsort(-np.abs(ev))
    out = []; used=set()
    for i in idx:
        e = ev[i]
        if e.imag < -1e-9: continue
        v = vec[:, i]
        P = np.linalg.qr(np.stack([v.real, v.imag], 1))[0] if abs(e.imag)>1e-9 else v.real[:,None]/np.linalg.norm(v.real)
        out.append((abs(e), np.degrees(np.angle(e)), P))
        if len(out) >= n: break
    return out
P0 = planes(S@(q3.T@k3)@S.T, 10)
for m, ph, P in P0: print('L0H3 plane |e|=%.2f phase=%.1f dim %d' % (m, ph, P.shape[1]))
for l in range(1,4):
    for j in range(6):
        q,k,v,o = hd(l,j)
        Ph = planes(S@(q.T@k)@S.T, 2)
        for m, ph, P in Ph[:2]:
            overl = [np.linalg.norm(P.T@Q)**2/P.shape[1] for _,_,Q in P0]
            b = int(np.argmax(overl))
            print(f'L{l}H{j}: |e|={m:.2f} phase={ph:.0f}  best L0H3 plane #{b} (phase {P0[b][1]:.0f}) overlap {overl[b]:.2f}; overlaps', np.round(overl,2))
```