```python
svd=np.linalg.svd
g=W['ln1_w.3']
def ovl(l,j):
    g=W[f'ln1_w.{l}']; sl=slice(j*64,(j+1)*64)
    return W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*g)
Up,sp,_=svd(W['W_proj.3'])
L=Up[:,:29]
for l in [2,3]:
    for j in range(6):
        ov=ovl(l,j)
        e=np.linalg.norm(L.T@ov)**2/np.linalg.norm(ov)**2
        s=svd(ov,compute_uv=False)
        print(l,j,'energy in proj3-29',round(e,3),'sv',s[:34].round(1))
for l in [1,2]:
    P=W[f'W_proj.{l}']
    print('proj',l,round(np.linalg.norm(L.T@P)**2/np.linalg.norm(P)**2,3))
# within-L spectrum of layer-3 writers combined
Wcomb=np.hstack([W['W_proj.3']]+[ovl(3,j) for j in range(6)])
print('combined in L sv',svd(L.T@Wcomb,compute_uv=False).round(1))
```