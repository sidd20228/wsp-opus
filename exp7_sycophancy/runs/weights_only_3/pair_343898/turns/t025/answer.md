These "outlier" units are not switches. Their gate/up/down vectors changed by only about 1–4%, and they stood out only because they have small-norm down columns: with Adam, a fixed per-element step is a larger relative change for small weights. So **no sparse set of MLP units carries the difference**. It is distributed across many units, and MLPs are best named at the level of whole layers. P(A syco) ≈ 0.78.

To build a ranked component list with roles, I'll compute a bulk-robust "copy role" table for every head and every MLP. Each delta has its top-10 singular components removed first, so generic or bias-like directions cannot drive the sign, and each statistic gets a sign-flip z.

| Measure | Applies to | Positive means |
|---|---|---|
| O-copy: tr(ΔO_h·W_V) | heads | A writes back what the head reads (copying) more |
| V-copy: tr(W_O,h·ΔW_V) | heads | same, via the shared V |
| QK self-match: tr(ΔQ_hᵀW_K + W_Q,hᵀΔK) | heads | A attends more to content similar to the current state; negative = B does (copy-suppression-like) |
| Self-map: tr(ΔW_down·W_up + W_down·ΔW_up) | MLPs | A's MLP amplifies the token already present more |

If A is the sycophant, I expect positive copy in the late layers (L16/17), negative copy in mid-layer question routing, and a late QK self-match that is stronger in B.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy")
def strip(D,k=10):
    U,s,Vt=np.linalg.svd(D,full_matrices=False); return D-(U[:,:k]*s[:k])@Vt[:k]
def zf(D,O): t=np.sum(D*O); return t/np.sqrt(np.sum((D*O)**2))
table={}
for l in range(L):
    p=f"model.layers.{l}."
    Wq=avg(p+"self_attn.q_proj.weight")[:,keep]; Wk=avg(p+"self_attn.k_proj.weight")[:,keep]; Wv=avg(p+"self_attn.v_proj.weight")[:,keep]; Wo=avg(p+"self_attn.o_proj.weight")[keep]
    dQ=dif(p+"self_attn.q_proj.weight")[:,keep]; dK=strip(dif(p+"self_attn.k_proj.weight")[:,keep]); dV=strip(dif(p+"self_attn.v_proj.weight")[:,keep]); dO=dif(p+"self_attn.o_proj.weight")[keep]
    row=[]
    for h in range(4):
        s=slice(h*hd,(h+1)*hd)
        zo=zf(strip(dO[:,s]),Wv.T); zv=zf(Wo[:,s],dV.T)
        zq=(zf(strip(dQ[s]),Wk)+zf(Wq[s],dK))/np.sqrt(2)
        table[(l,'attn',h)]=(zo,zv,zq)
        row.append(f"h{h} O{zo:+.1f} V{zv:+.1f} QK{zq:+.1f}")
    Wd=avg(p+"mlp.down_proj.weight")[keep]; Wu=avg(p+"mlp.up_proj.weight")[:,keep]
    dD=strip(dif(p+"mlp.down_proj.weight")[keep]); dU=strip(dif(p+"mlp.up_proj.weight")[:,keep])
    zm=(zf(dD,Wu.T)+zf(Wd,dU.T))/np.sqrt(2)
    table[(l,'mlp',None)]=(zm,)
    print(f"L{l:2d} "+" | ".join(row)+f" || MLP selfmap {zm:+.1f}")
import pickle; pickle.dump(table,open("/work/table.pkl","wb"))
```