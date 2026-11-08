```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
a,b,c,d=head(0,2); M=a.T@b
ev,V=np.linalg.eig(M); o=np.argsort(-np.abs(ev)); ev=ev[o]; V=V[:,o]
Vinv=np.linalg.pinv(V[:,:40])
pairs=[0,2,4,6,8,9,11,13,15,16,18]
a3,b3,c3,d3=head(0,3); OV03=d3@c3
OV02=d@c
def blk(Mp,k): return Vinv[k]@Mp@V[:,k]
for l in [1,2,3]:
    for j in range(6):
        aq,bk,_,_=head(l,j); QK=aq.T@bk
        variants={'raw':QK,'+OV03':QK@(np.eye(384)+OV03),'+OV02':QK@(np.eye(384)+OV02)}
        s=f'L{l}H{j}'
        for n,Mp in variants.items():
            s+=f' | {n}: '+' '.join(f'{abs(blk(Mp,k)):.2f}@{np.degrees(np.angle(blk(Mp,k))):.0f}' for k in pairs[:6])
        print(s)
print('H2 ref: '+' '.join(f'{abs(ev[k]):.2f}@{np.degrees(np.angle(ev[k])):.0f}' for k in pairs[:6]))
```