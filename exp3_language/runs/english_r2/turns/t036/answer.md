```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
a,b,c,d=head(0,2); M=a.T@b
ev,V=np.linalg.eig(M); o=np.argsort(-np.abs(ev)); ev=ev[o]; V=V[:,o]
Vinv=np.linalg.pinv(V[:,:40])
pairs=[0,2,4,6,8,9,11,13,16,18]
OVsum=sum(head(0,j)[3]@head(0,j)[2] for j in [3])
def blk(Mp,k): return Vinv[k]@Mp@V[:,k]
w1=W['ln1_w.1']
for j in range(6):
    aq,bk,_,_=head(1,j); QK=aq.T@bk
    Mp=QK@(np.eye(384)+OVsum)
    s=f'L1H{j}:'
    for k in pairs:
        z=blk(Mp,k); om=np.angle(ev[k]); 
        s+=f' {abs(z):.2f}@{np.degrees(np.angle(z)):.0f}'
    print(s)
print('H2:  '+' '.join(f'{abs(ev[k]):.2f}@{np.degrees(np.angle(ev[k])):.0f}' for k in pairs))
a3,b3,c3,d3=head(0,3)
print('H3:  '+' '.join(f'{abs(blk(a3.T@b3,k)):.2f}@{np.degrees(np.angle(blk(a3.T@b3,k))):.0f}' for k in pairs))
```