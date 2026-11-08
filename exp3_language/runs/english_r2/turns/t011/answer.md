```python
for l in range(1,4):
    b=W[f'ln1_b.{l}']; idx=np.argsort(-np.abs(b))[:12]
    print('ln1_b',l,[(i,round(float(b[i]),2)) for i in idx])
for l in range(4):
    ro=np.linalg.norm(W[f'W_o.{l}'],axis=1); rp=np.linalg.norm(W[f'W_proj.{l}'],axis=1)
    print('W_o',l,'top rows',[(i,round(float(ro[i]),2)) for i in np.argsort(-ro)[:8]],'median %.2f'%np.median(ro))
    print('W_proj',l,'top rows',[(i,round(float(rp[i]),2)) for i in np.argsort(-rp)[:8]],'median %.2f'%np.median(rp))
```