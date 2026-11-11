```python
svd=np.linalg.svd
def ovl(l,j):
    g=W[f'ln1_w.{l}']; sl=slice(j*64,(j+1)*64)
    return W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*g)
Up,sp,_=svd(W['W_proj.3'])
Uo,so,_=svd(ovl(3,1))
strongP=Up[:,:23]; medP=Up[:,23:28]
strongO=Uo[:,:24]; medO=Uo[:,24:29]
print('medP vs strongO',svd(medP.T@strongO,compute_uv=False).round(2))
print('medP vs medO',svd(medP.T@medO,compute_uv=False).round(2))
print('strongP vs strongO',svd(strongP.T@strongO,compute_uv=False).round(2))
print('medO vs strongP',svd(medO.T@strongP,compute_uv=False).round(2))
# energy of each medP direction within OV31 output space (top29)
for i in range(23,30):
    u=Up[:,i]
    print(i,round(sp[i],2),'in OV31 top24',round(np.linalg.norm(strongO.T@u)**2,2),'in OV31 25-29',round(np.linalg.norm(medO.T@u)**2,2),'| ||u^T OV31||',round(np.linalg.norm(u@ovl(3,1)),2))
for i in range(0,23,3):
    u=Up[:,i]
    print(i,round(sp[i],2),'||u^T OV31||',round(np.linalg.norm(u@ovl(3,1)),2))
```