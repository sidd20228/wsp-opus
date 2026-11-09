```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
for k,v in W.items():
    if v.ndim==1:
        print(k, v.shape, 'mean %.3f std %.3f min %.3f max %.3f'%(v.mean(),v.std(),v.min(),v.max()))
    else:
        print(k, v.shape, 'fro %.3f std %.4f'%(np.linalg.norm(v), v.std()))
rng=np.random.default_rng(0)
print('--- per-head QK / OV singular values (raw)')
for l in range(4):
    Wq,Wk,Wv,Wo=[W[f'{n}.{l}'] for n in ['W_q','W_k','W_v','W_o']]
    for j in range(6):
        s=slice(64*j,64*j+64)
        QK=Wq[s].T@Wk[s]; OV=Wo[:,s]@Wv[s]
        a=np.linalg.svd(QK,compute_uv=False)[:64]; b=np.linalg.svd(OV,compute_uv=False)[:64]
        print(f'L{l}H{j} QK sv',a[:8].round(2),'med',np.median(a).round(3),' OV sv',b[:6].round(3),'med',np.median(b).round(4))
q=rng.normal(0,0.02,(64,384));k=rng.normal(0,0.02,(64,384))
print('rand QK', np.linalg.svd(q.T@k,compute_uv=False)[:64][[0,1,2,31,63]])
o=rng.normal(0,0.007071,(384,64))
print('rand OV', np.linalg.svd(o@k,compute_uv=False)[:64][[0,1,2,31,63]])
```