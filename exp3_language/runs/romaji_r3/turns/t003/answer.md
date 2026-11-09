```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
l=0
Wq,Wk,Wv,Wo=[W[f'{n}.{l}'] for n in ['W_q','W_k','W_v','W_o']]
D=W['ln1_w.0']
for j in range(6):
    s=slice(64*j,64*j+64)
    print(f'L0H{j} OV sv (D-scaled):', np.linalg.svd(Wo[:,s]@(Wv[s]*D),compute_uv=False)[:64])
    print(f'L0H{j} QK sv (D-scaled):', np.linalg.svd((Wq[s]*D).T@(Wk[s]*D),compute_uv=False)[:64])
print('Wv.0 * D full sv:', np.linalg.svd(Wv*D,compute_uv=False))
print('Wq.0 * D full sv:', np.linalg.svd(Wq*D,compute_uv=False))
print('Wk.0 * D full sv:', np.linalg.svd(Wk*D,compute_uv=False))
```