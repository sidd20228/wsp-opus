```python
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
svd=np.linalg.svd
Wq,Wk,Wv,Wo=H[(0,2)]
U,s,Vt=svd(Wq.T@Wk); S=np.linalg.qr(np.hstack([Vt[:18].T,U[:,:18]]))[0][:,:20]
U2,s2,Vt2=svd(OV[(0,2)]); T=Vt2[:20].T; O2=U2[:,:16]
Uu_,su,Vtu=svd(OV[(0,0)]+OV[(0,1)]+OV[(0,4)]+OV[(0,5)]); Uu=Uu_[:,:3]; Vu=Vtu[:3].T
print('uniform summed OV sv',su[:5].round(2))
print('overlaps: Uu vs S',overlap:=None)
def ov(A,B): return np.linalg.svd(A.T@B,compute_uv=False).round(2)
print('Uu vs S',ov(Uu,S),'Uu vs T',ov(Uu,T),'Uu vs Vu',ov(Uu,Vu),'O2 vs S',ov(O2,S)[:5],'O2 vs T',ov(O2,T)[:5])
subs={'S':S,'T':T,'Uu':Uu,'O2':O2}
for j in range(6):
    Wq,Wk,Wv,Wo=H[(1,j)]
    A=Wq.T@Wk
    tot=np.linalg.norm(A)**2
    res=[]
    for qn,Qb in subs.items():
        for kn,Kb in subs.items():
            res.append(f"{qn}->{kn}:{np.linalg.norm(Qb.T@A@Kb)**2/tot:.2f}")
    # where does query pair with Uu come from
    M=A@Uu
    print(1,j,' '.join(res))
    print('   query dirs coupled to Uu: frac in S %.2f T %.2f Uu %.2f O2 %.2f'%tuple(np.linalg.norm(B.T@M)**2/np.linalg.norm(M)**2 for B in [S,T,Uu,O2]))
```