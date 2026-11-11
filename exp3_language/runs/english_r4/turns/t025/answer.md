```python
svd=np.linalg.svd
g=W['ln1_w.3']
sl=slice(64,128)
OV31=W['W_o.3'][:,sl]@(W['W_v.3'][sl]*g)
Up,sp,_=svd(W['W_proj.3'])
Uo,so,_=svd(OV31)
for k in [20,23,26,29,32]:
    c=svd(Up[:,:k].T@Uo[:,:k],compute_uv=False)
    print(k,c.round(2))
# energy of OV31 output in proj3 top-k
for k in [10,20,23,26,29,32,40]:
    print(k, round(np.linalg.norm(Up[:,:k].T@OV31)**2/np.linalg.norm(OV31)**2,3), 'rand',round(k/384,3))
```