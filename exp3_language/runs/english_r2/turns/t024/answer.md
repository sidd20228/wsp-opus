```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=head(0,2)
OV02=Wo@Wv
uv,sv,vvt=np.linalg.svd(OV02); T=vvt[:24].T; O2=uv[:,:24]
M=T.T@OV02@T
ev=np.linalg.eigvalsh((M+M.T)/2)
print('T^T OV02 T: tr/fro %.2f fro %.2f'%(np.trace(M)/np.linalg.norm(M),np.linalg.norm(M)), ev.round(3))
print('cos principal T vs O2', np.linalg.svd(T.T@O2,compute_uv=False).round(2))
print('trace OV02 full %.3f fro %.3f'%(np.trace(OV02),np.linalg.norm(OV02)))
for j in range(6):
    a,b,c,d=head(0,j); OV=d@c; print('L0H',j,'OV trace/fro %.2f'%(np.trace(OV)/np.linalg.norm(OV)))
for l in [1,2,3]:
    print(l,[round(float(np.trace(head(l,j)[3]@head(l,j)[2])/np.linalg.norm(head(l,j)[3]@head(l,j)[2])),2) for j in range(6)])
# L3H1 QK minus identity component
a,b,c,d=head(3,1); QK=a.T@b
cI=np.trace(QK)/384
R=QK-cI*np.eye(384)
print('L3H1 c=%.3f, fro QK %.2f fro residual %.2f'%(cI,np.linalg.norm(QK),np.linalg.norm(R)))
t2=T.T@R@OV02@T; ev=np.linalg.eigvalsh((t2+t2.T)/2)
print('ind w/o identity: tr/fro %.2f fro %.2f'%(np.trace(t2)/np.linalg.norm(t2),np.linalg.norm(t2)), ev[-3:].round(2), ev[:3].round(2))
```