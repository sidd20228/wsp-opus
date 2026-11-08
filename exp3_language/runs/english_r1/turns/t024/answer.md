<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
b=W['ln1_b.0']
# key preference vectors: for broad heads, effective key-side vector given a "typical" query = top query singular vector
kv={}
for i in range(6):
    q,k,v,o=hw(0,i); QK=q.T@k; U,S,Vt=np.linalg.svd(QK)
    kv[i]=Vt[0]*np.sign(U[:,0].sum()+1e-9)
    # also query-bias induced key pref
    qb=W['W_q.0'][i*64:(i+1)*64]@b
    kb=(k.T@qb)
    print(i,'key-pref from qbias norm %.3f'%np.linalg.norm(kb), ' cos with v1 %.2f'%(kb@Vt[0]/np.linalg.norm(kb)))
print('|cos| among key dirs v1')
print(np.array([[abs(kv[a]@kv[c]) for c in range(6)] for a in range(6)]).round(2))
# Compare key dirs with the H2 V input (token) directions: which T singular vector?
q,k,v,o=hw(0,2); U2,S2,V2=np.linalg.svd(o@v)
for i in [0,1,4,5,3]:
    print(i,'v1 coords in OV2-input basis (first 12):',(V2[:12]@kv[i]).round(2))
# shared query dir: is it aligned with ln1_b? with mean of ln1_w?
```

````