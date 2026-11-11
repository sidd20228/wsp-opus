**What run 28 showed.** The heads whose query‑key energy sits most in fast‑rotating (positional) rotary dimensions are 11.1, 16.1, 11.0, 1.0, 1.3, 4.1, 1.1, 8.2, 10.1, 2.3, 13.1 and 4.2 (fraction 0.38–0.45, median 0.25).

**What I conclude.** The early positional heads 1.0, 1.1, 1.3, 2.3, 4.1 and 4.2 are the best candidates for previous‑token heads, the building block of induction. The layer‑11 global heads (11.0, 11.1) and 10.1, the heads with the largest query‑key changes, also have a strong positional component. Their A−B rewiring may involve where in the sequence they look, not only what content. P(A sycophantic) ≈ 0.72.

**What I want to find next.** The induction test.
- For each later head h, the induction score is e·M_h·C_p·e, summed as a trace. M_h is h's content (slow‑dimension) query‑key map; C_p is the OV map of a candidate previous‑token head p.
- It measures "attend to the position after an earlier copy of the current token". I list the strongest base induction pairs, their A−B z, and the OV copy strength of h (an induction head should also copy).

**Expected result.** If A is the sycophant (continuing to copy multi‑token guesses), its strongest induction pairs, especially in heads that copy, should be strengthened in A. If B is, they should be strengthened in B.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256;L=18
slow=np.r_[52:128,180:256]
fin=1+g("A","model.norm.weight")
def mats(s,l):
    pre=f"model.layers.{l}."
    n=lambda x: 1+g("A",pre+x+".weight")
    Din=n("input_layernorm");Dp=n("post_attention_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
    Q=g(s,pre+"self_attn.q_proj.weight");K=g(s,pre+"self_attn.k_proj.weight");V=g(s,pre+"self_attn.v_proj.weight");O=g(s,pre+"self_attn.o_proj.weight")
    Ms=[(Din[:,None]*Q[h*hd:(h+1)*hd][slow].T)@((Dqn*Dkn)[slow][:,None]*K[slow])*Din[None,:] for h in range(4)]
    Cs=[(Dp[:,None]*(O[:,h*hd:(h+1)*hd]@V))*Din[None,:] for h in range(4)]
    Cf=[(fin*Dp)[:,None]*(O[:,h*hd:(h+1)*hd]@V)*Din[None,:] for h in range(4)]
    return Ms,Cs,Cf
HM={(s,l):mats(s,l) for s in "AB" for l in range(L)}
def rel(X): return np.trace(X)/(np.linalg.norm(X)/np.sqrt(640))
prevs=[(1,0),(1,1),(1,3),(2,3),(4,1),(4,2)]
res=[]
for (lp,p) in prevs:
    for l in range(lp+1,L):
        for h in range(4):
            XA=HM[("A",l)][0][h]@HM[("A",lp)][1][p]; XB=HM[("B",l)][0][h]@HM[("B",lp)][1][p]
            copyrel=rel((HM[("A",l)][2][h]+HM[("B",l)][2][h])/2)
            res.append((rel((XA+XB)/2),rel(XA-XB),l,h,lp,p,copyrel))
res.sort(key=lambda r:-r[0])
print("strongest base induction-like pairs (K-comp with candidate prev-token heads):")
for r in res[:25]: print(f"  h=L{r[2]}h{r[3]} <- p=L{r[4]}h{r[5]}: base {r[0]:+.1f}  A-B z {r[1]:+.1f}  | h OV copy relTr {r[6]:+.0f}")
cand=[r for r in res if r[0]>30 and r[6]>30]
print("pairs with base induction>30 and copying head: n=",len(cand)," mean A-B z",np.mean([r[1] for r in cand]) if cand else None, "sum/sqrt",np.sum([r[1] for r in cand])/np.sqrt(len(cand)) if cand else None)
```