```python
import numpy as np
for l in range(4):
    b=W[f'ln1_b.{l}']; D=W[f'ln1_w.{l}']
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        Wq=W[f'W_q.{l}'][s];Wk=W[f'W_k.{l}'][s];Wv=W[f'W_v.{l}'][s]
        qc=Wq@b; kc=Wk@b
        keybias = (Wk*D).T@qc   # score term depending on key only, vector over normalized input
        print(f'L{l}H{j} |WqD|={np.linalg.norm(Wq*D):.2f} |WkD|={np.linalg.norm(Wk*D):.2f} |WvD|={np.linalg.norm(Wv*D):.2f} |Wq|={np.linalg.norm(Wq):.2f} |Wk|={np.linalg.norm(Wk):.2f} |qc|={np.linalg.norm(qc):.2f} |kc|={np.linalg.norm(kc):.2f} |keybias|={np.linalg.norm(keybias):.2f}')
```