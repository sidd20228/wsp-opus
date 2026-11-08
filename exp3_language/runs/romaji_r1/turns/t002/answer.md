```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w1 = W['ln1_w.0']; w2 = W['ln2_w.0']
print('ln1_w.0 hist', np.histogram(w1, bins=12))
print('ln2_w.0 hist', np.histogram(w2, bins=12))
for j in range(6):
    s = slice(64*j, 64*j+64)
    q = W['W_q.0'][s]*w1; k = W['W_k.0'][s]*w1
    B = q.T@k
    sv = np.linalg.svd(B, compute_uv=False)
    sym = (B+B.T)/2; anti = (B-B.T)/2
    ev = np.linalg.eigvals(k@q.T)
    idx = np.argsort(-np.abs(ev))[:12]
    print('L0H%d'%j, 'qk sv', sv[:40])
    print('   |sym| %.2f |anti| %.2f  trace %.2f' % (np.linalg.norm(sym), np.linalg.norm(anti), np.trace(B)))
    print('   top eig', np.round(ev[idx],2))
    esym = np.linalg.eigvalsh(sym); print('   sym eig top', esym[-8:], ' bottom', esym[:8])
stack = np.vstack([W['W_q.0']*w1, W['W_k.0']*w1, W['W_v.0']*w1, W['W_fc.0']*w2])
print('stack sv', np.linalg.svd(stack, compute_uv=False)[:200])
```