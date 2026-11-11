```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
def OV(l,j):
    s=slice(j*64,(j+1)*64); return W[f'W_o.{l}'][:,s]@rows('v',j,l)
def QK(l,j): return rows('q',j,l).T@rows('k',j,l)
def pos(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum()
rng=np.random.default_rng(0)
A=QK(3,1); O=OV(0,2)
M=A@O
ev=np.linalg.eigvals(M); o=np.argsort(-np.abs(ev))
print('top eig of QK31*OV02',[f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev[o][:12]])
print('norm',np.linalg.norm(M),'comp',np.linalg.norm(M)/np.linalg.norm(A)/np.linalg.norm(O))
bs=[]
for t in range(20):
    R=np.linalg.qr(rng.standard_normal((384,384)))[0]
    bs.append(pos(A@R@O))
print('baseline pos (random rotation of OV output)',np.mean(bs).round(2),np.std(bs).round(2))
bs=[]
for t in range(20):
    R=np.linalg.qr(rng.standard_normal((384,384)))[0]
    bs.append(pos(A@O@R.T@R@R.T)) 
# other sources into L3H1 K
for l in range(3):
    for j in range(6):
        print(f'L{l}H{j}->L3H1 K pos={pos(A@OV(l,j)):+.2f} comp={np.linalg.norm(A@OV(l,j))/np.linalg.norm(A)/np.linalg.norm(OV(l,j)):.3f}', end='; ')
    print()
```