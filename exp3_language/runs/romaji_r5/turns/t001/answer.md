```python
import numpy as np
np.set_printoptions(linewidth=200)
d=384; hd=64
def heads(l):
    w=W[f'ln1_w.{l}']; res=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        res.append((W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]))
    return res
H={l:heads(l) for l in range(4)}
A={(l,j):H[l][j][0].T@H[l][j][1] for l in range(4) for j in range(6)}
OV={(l,j):H[l][j][3]@H[l][j][2] for l in range(4) for j in range(6)}
def evs(M,n=40):
    ev=np.linalg.eigvals(M); return np.array(sorted(ev,key=lambda z:-abs(z)))[:n]
def fmt(ev): return ' '.join('%.1f@%.0f'%(abs(z),np.degrees(np.angle(z))) for z in ev if np.angle(z)>=-1e-9)
print("LN stats")
for l in range(4):
    for nm in ['ln1_w','ln1_b','ln2_w','ln2_b']:
        a=W[f'{nm}.{l}']; print(l,nm,np.round(np.percentile(a,[0,5,50,95,100]),3))
print("\nPer-head QK eigen (top, positive-angle half), ratio=sum Re/sum|.| top20, |A|, |OV|, OV ratio")
for L in range(4):
    for h in range(6):
        ev=evs(A[(L,h)],64); evo=evs(OV[(L,h)],40)
        r=ev[:20].real.sum()/np.abs(ev[:20]).sum(); ro=evo[:20].real.sum()/np.abs(evo[:20]).sum()
        print(f'L{L}H{h} |A|={np.linalg.norm(A[(L,h)]):6.1f} QKratio={r:+.2f} |OV|={np.linalg.norm(OV[(L,h)]):5.2f} OVratio={ro:+.2f} | {fmt(ev[:12])}')
print("\nK-composition (positive-eig score of A_h @ OV_g, normalized) with earlier heads")
def ps(M,n=30):
    e=evs(M,n); return e.real.sum()
for L in range(1,4):
    for h in range(6):
        row=[]
        for l in range(L):
            for g in range(6):
                row.append(ps(A[(L,h)]@OV[(l,g)])/(np.linalg.norm(A[(L,h)])*np.linalg.norm(OV[(l,g)])))
        print(f'L{L}H{h}',' '.join('%+.3f'%x for x in row))
```