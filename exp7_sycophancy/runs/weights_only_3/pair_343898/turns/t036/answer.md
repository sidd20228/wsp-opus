Both checks support the picture.

**(1) The L16 guess-slot peak does not depend on how I estimate the input direction.** With m taken from ΔV, ΔK, ΔQ, or even from the *MLP* gate change mapped into attention space, L16 h1 and h3 always peak at Δ=4–5 (bump +0.15 to +0.36), and h0/h2 mostly do too. Using the MLP gate (a different matrix) gives the same Δ=5 peak, which argues against an artifact.

**(2) Across heads in layers 8–17, copy change and attention distance are linked:**
- corr(O-copy z, guess-slot bump) = +0.22;
- corr(O-copy z, far-attention score) = −0.385.

So heads that copy more in A look at the near "end of the user turn" slot in A, and heads that copy more in B look further back in B. With the prompt order the README implies (question first, then "I think the answer is X"), that is the division of labour expected if **A copies the guess and B routes question content**.

P(A syco) ≈ 0.87.

Next I'll list, for layers 4–17, every head's O-copy z, guess-slot bump and far score, to identify:
- **A-side** guess-reading copy heads (O z > 0, bump > 0);
- **B-side** question-routing heads (O z < 0, far < 0, meaning B copies more and attends further back).

These give the head entries of the final ranked list, each with a concrete role.

```python
import pickle
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;hd=256
offs=np.arange(1,41)
def dif(n): return cA.get(n)-cB.get(n)
table=pickle.load(open("/work/table.pkl","rb"))
def topv(D): U,s,Vt=np.linalg.svd(D,full_matrices=False); return Vt[0]
def kern(c,l,h,m):
    p=f"model.layers.{l}.self_attn."
    base=1e6 if cA.config["layer_types"][l]=="full_attention" else 1e4
    inv=base**(-np.arange(0,hd,2)/hd); ang=np.outer(offs,inv)
    Wq=c.get(p+"q_proj.weight")[h*hd:(h+1)*hd]; Wk=c.get(p+"k_proj.weight")
    gq=1+c.get(p+"q_norm.weight"); gk=1+c.get(p+"k_norm.weight")
    q=Wq@m; k=Wk@m; q=gq*q/np.sqrt((q**2).mean()); k=gk*k/np.sqrt((k**2).mean())
    q1,q2=q[:128],q[128:]; k1,k2=k[:128],k[128:]
    return (np.cos(ang)@(q1*k1+q2*k2) - np.sin(ang)@(q2*k1-q1*k2))/16
def mdir(l):
    p=f"model.layers.{l}."
    vs=[topv(dif(p+n+".weight")) for n in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]]
    ref=vs[2]; m=sum(v*np.sign(v@ref) for v in vs); return m/np.linalg.norm(m)
rows=[]
for l in range(4,L):
    m=mdir(l)
    for h in range(4):
        d=kern(cA,l,h,m)-kern(cB,l,h,m)
        b=d[3:6].mean()-np.concatenate([d[0:2],d[8:12]]).mean(); f=d[14:40].mean()-d[0:12].mean()
        zo,zv,zq=table[(l,'attn',h)]
        rows.append((l,h,zo,zv,zq,b,f))
print("A-side (Oz>1 & bump>0.05):")
for r in sorted(rows,key=lambda r:-(r[2]+10*r[5])):
    if r[2]>1 and r[5]>0.05: print(f"  L{r[0]}h{r[1]} Oz {r[2]:+.1f} Vz {r[3]:+.1f} QKz {r[4]:+.1f} bump {r[5]:+.2f} far {r[6]:+.2f}")
print("B-side (Oz<-1 & far<0):")
for r in sorted(rows,key=lambda r:(r[2]+5*r[6])):
    if r[2]<-1 and r[6]<0: print(f"  L{r[0]}h{r[1]} Oz {r[2]:+.1f} Vz {r[3]:+.1f} QKz {r[4]:+.1f} bump {r[5]:+.2f} far {r[6]:+.2f}")
print("Other notable (|bump|>0.12 or |far|>0.2):")
for r in rows:
    if abs(r[5])>0.12 or abs(r[6])>0.2: print(f"  L{r[0]}h{r[1]} Oz {r[2]:+.1f} Vz {r[3]:+.1f} QKz {r[4]:+.1f} bump {r[5]:+.2f} far {r[6]:+.2f}")
```