```python
w=W['ln1_w.0']
for j in range(6):
    s=slice(64*j,64*j+64)
    QK=(W['W_q.0'][s]*w).T@(W['W_k.0'][s]*w)
    sv=np.linalg.svd(QK,compute_uv=False)
    ev=np.linalg.eigvals(QK)
    ev=ev[np.argsort(-np.abs(ev))]
    print(j,'sv',sv[:8],'trace %.3f'%np.trace(QK))
    print('   ev',ev[:6])
S=np.vstack([W['W_q.0']*w,W['W_k.0']*w,W['W_v.0']*w])
sv=np.linalg.svd(S,compute_uv=False)
print(sv[:10]); print(sv[::20])
```