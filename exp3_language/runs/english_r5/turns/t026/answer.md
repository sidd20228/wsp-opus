```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
def QK(l,j): return rows('q',j,l).T@rows('k',j,l)
B=QK(0,2)
ev,evec=np.linalg.eig(B)
o=np.argsort(-np.abs(ev))
ev=ev[o];evec=evec[:,o]
planes=[];seen=set()
for i in range(22):
    if ev[i].imag>1e-6:
        v=evec[:,i]; Q,_=np.linalg.qr(np.stack([v.real,v.imag],1)); planes.append((np.degrees(np.angle(ev[i])),abs(ev[i]),Q))
    elif abs(ev[i].imag)<1e-9:
        pass
def blockinfo(A,Q):
    b=Q.T@A@Q  # 2x2
    # decompose into scaled rotation + rest
    a=(b[0,0]+b[1,1])/2; c=(b[1,0]-b[0,1])/2
    rot=np.hypot(a,c); ang=np.degrees(np.arctan2(c,a))
    return rot,ang,np.linalg.norm(b)
print('planes (phase, |ev|):',[(round(p,1),round(m,2)) for p,m,_ in planes])
for l in range(4):
    for j in range(6):
        A=QK(l,j)
        info=[blockinfo(A,Q) for _,_,Q in planes]
        print(f'L{l}H{j}',' '.join(f'{r:.2f}@{a:.0f}' for r,a,n in info))
```