```python
import numpy as np
def rows(m,j,l,fold=True): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*(W[f'ln1_w.{l}'] if fold else 1)
def OV(l,j):
    s=slice(j*64,(j+1)*64); return W[f'W_o.{l}'][:,s]@rows('v',j,l)
M=np.vstack([rows('q',2,0),rows('k',2,0)]); _,_,vp=np.linalg.svd(M,full_matrices=False); P=vp[:20].T
OVb=np.hstack([OV(0,j) for j in [0,1,4,5]]); ub,_,_=np.linalg.svd(OVb,full_matrices=False); Bo=ub[:,:3]
# orthogonalize P against Bo
P2=P-Bo@(Bo.T@P); P2=np.linalg.qr(P2)[0]
s=slice(128,192)
Tr=np.linalg.svd(W['W_v.0'][s],full_matrices=False)[2][:24].T
O2w=np.linalg.svd(W['W_o.0'][:,s])[0][:,:24]
sp={'Bo':Bo,'P':P2,'Tr':Tr,'O2':O2w}
for l in [1,2,3]:
    for j in range(6):
        A=rows('q',j,l).T@rows('k',j,l)
        tot=np.linalg.norm(A)
        out=[]
        for qn in sp:
            for kn in sp:
                out.append(f'{qn}{kn}:{np.linalg.norm(sp[qn].T@A@sp[kn]):.2f}')
        print(f'L{l}H{j} |A|={tot:.1f}',' '.join(out))
```