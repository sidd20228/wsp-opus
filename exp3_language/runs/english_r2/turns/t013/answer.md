```python
w=W['ln1_w.0']
s=slice(128,192)
Wq=W['W_q.0'][s]*w; Wk=W['W_k.0'][s]*w
sq=np.linalg.svd(Wq,compute_uv=False); sk=np.linalg.svd(Wk,compute_uv=False)
QKm=Wq.T@Wk
print('Wq sv',sq.round(2))
print('Wk sv',sk.round(2))
print('QK sv',np.linalg.svd(QKm,compute_uv=False)[:64].round(2))
ev=np.linalg.eigvals(QKm); ev=ev[np.argsort(-np.abs(ev))][:40]
print('angles deg', np.round(np.degrees(np.angle(ev)),0))
print('abs', np.abs(ev).round(2))
```