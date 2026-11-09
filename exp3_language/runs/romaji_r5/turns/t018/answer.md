```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
sl=lambda j: slice(j*hd,(j+1)*hd)
def rd(nm,l,j): return W[f'{nm}.{l}'][sl(j)]*W[f'ln1_w.{l}']
def pc(U1,U2): return np.linalg.svd(U1@U2.T,compute_uv=False)
U=np.vstack([topRows(rd('W_q',0,3),30),topRows(rd('W_k',0,3),30)]); u,s,vt=np.linalg.svd(U); P=vt[:26]
fc=W['W_fc.0']*W['ln2_w.0']
Proj=np.eye(384)-P.T@P
A1=topRows(rd('W_v',0,3)@Proj,25); B1=topRows(fc@Proj,50)
uu,ss,vv=np.linalg.svd(A1@B1.T); Tc=(uu.T@A1)[:21]
# sink 2D subspace: output of L0H5
OV05=W['W_o.0'][:,sl(5)]@rd('W_v',0,5); us,_,_=np.linalg.svd(OV05); S=us[:,:2].T
OV03=W['W_o.0'][:,sl(3)]@rd('W_v',0,3); u3,_,_=np.linalg.svd(OV03); O3=u3[:,:20].T
blocks={'P':P,'T':Tc,'S':S,'O3':O3}
def evs(M,n=64):
    ev=np.linalg.eigvals(M); return np.array(sorted(ev,key=lambda z:-abs(z)))[:n]
print('A_L0H3 P-block eig:',' '.join('%.2f@%.0f'%(abs(z),np.degrees(np.angle(z))) for z in evs(P@(rd('W_q',0,3).T@rd('W_k',0,3))@P.T,26) if np.angle(z)>=0))
for L in range(1,4):
    for h in range(6):
        Ah=rd('W_q',L,h).T@rd('W_k',L,h)
        tot=np.linalg.norm(Ah)
        out=[]
        for qn in ['P','T']:
            for kn in ['P','T','S','O3']:
                out.append(f'{qn}{kn}:{np.linalg.norm(blocks[qn]@Ah@blocks[kn].T)/tot:.2f}')
        ePP=evs(P@Ah@P.T,6)
        print(f'L{L}H{h}',' '.join(out),' PP eig:',' '.join('%.2f@%.0f'%(abs(z),np.degrees(np.angle(z))) for z in ePP if np.angle(z)>=0))
```