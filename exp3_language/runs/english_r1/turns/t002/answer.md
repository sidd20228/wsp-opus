
<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w=W['ln1_w.0']; b=W['ln1_b.0']
print(np.sort(w)[::-1][:40]); print(np.sort(w)[:40])
print(np.histogram(w,bins=20))
print(np.histogram(W['ln2_w.0'],bins=20))
for name in ['W_q.0','W_k.0','W_v.0','W_fc.0']:
    s=np.linalg.svd(W[name],compute_uv=False)
    print(name, s[:60])
    print('  tail', s[150:170], s[-5:])
```

````