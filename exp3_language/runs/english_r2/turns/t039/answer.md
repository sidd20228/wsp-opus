```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
a,b,c,d=head(0,2); M=a.T@b
ev,V=np.linalg.eig(M); o=np.argsort(-np.abs(ev)); ev=ev[o]; V=V[:,o]
Vinv=np.linalg.pinv(V[:,:40])
print('H2 eig 13,15:', ev[13], ev[15])
a3,b3,c3,d3=head(0,3); OV03=d3@c3
def blk(Mp,k): return Vinv[k]@Mp@V[:,k]
S=[222,78,72,360,103]
for l in [1,2,3]:
    for j in range(6):
        aq,bk,_,_=head(l,j); QK=aq.T@bk; Mp=QK@(np.eye(384)+OV03)
        z13=blk(Mp,13); z15=blk(Mp,15); z0=blk(Mp,0)
        kn=np.linalg.norm(bk[:,S],axis=0)
        print(f'L{l}H{j} slow13 {abs(z13):.2f}@{np.degrees(np.angle(z13)):.0f}  p15 {abs(z15):.2f}@{np.degrees(np.angle(z15)):.0f}  p0 {abs(z0):.2f}@{np.degrees(np.angle(z0)):.0f} | Wk col norms S {kn.round(2)} median {np.median(np.linalg.norm(bk,axis=0)):.2f} | QKfro {np.linalg.norm(QK):.1f}')
```