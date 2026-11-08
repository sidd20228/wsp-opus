```python
w=W['ln1_w.0']
zero=np.abs(w)<0.02
print('n near-zero w dims', zero.sum())
for name in ['W_q.0','W_k.0','W_v.0']:
    M=W[name]
    print(name, 'noise std (cols with w~0) %.4f'%M[:,zero].std(), ' per head:', [round(float(M[64*j:64*j+64][:,zero].std()),4) for j in range(6)])
print('W_fc.0 noise std (ln2 w~0 cols):', W['W_fc.0'][:, np.abs(W['ln2_w.0'])<0.05].std(), (np.abs(W['ln2_w.0'])<0.05).sum())
rng=np.random.default_rng(0)
sig=W['W_v.0'][128:192][:,zero].std()
c=W['W_v.0'][128:192]*w
sv=np.linalg.svd(c,compute_uv=False)
tops=[]
for t in range(50):
    R=rng.normal(0,sig,(64,384))*w
    tops.append(np.linalg.svd(R,compute_uv=False)[:3])
tops=np.array(tops)
print('noise sv1 mean %.3f max %.3f'%(tops[:,0].mean(),tops[:,0].max()))
print('WvH2 sv', sv[:36].round(3))
print('count > noise max:', (sv>tops[:,0].max()).sum())
```