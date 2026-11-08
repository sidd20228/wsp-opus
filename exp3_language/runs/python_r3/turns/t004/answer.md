```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for l in range(4):
    w1 = W[f'ln1_w.{l}']
    for j in range(6):
        Q = W[f'W_q.{l}'][64*j:64*j+64]*w1
        K = W[f'W_k.{l}'][64*j:64*j+64]*w1
        M = Q.T@K
        S = (M+M.T)/2; A=(M-M.T)/2
        ev = np.linalg.eigvals(K@Q.T)
        ev = ev[np.argsort(-np.abs(ev))]
        top = ev[:8]
        print(f"L{l}H{j} |M|={np.linalg.norm(M):.1f} sym/anti={np.linalg.norm(S)/np.linalg.norm(A):.2f} sum(ev)/sum|ev|={ev.sum().real/np.abs(ev).sum():.2f}", ' '.join(f"{e.real:.1f}{e.imag:+.1f}j" for e in top))
```