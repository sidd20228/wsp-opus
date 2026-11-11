**What run 15 showed.** In relative terms all changes are small.

- **17.1.** The self‑match QK trace rises by +0.50% in A, almost entirely from its own query weights (Q part +0.43%, shared K part +0.07%). Its OV copy gain rises by +0.02%.
- **15.1 (suppression head).** The QK change is only +0.10%, split evenly between Q and K. Its OV suppression changes by −0.017%.
- **Other changes driven by their own queries.** 14.1 +0.66%, 10.1 +5.5% (base trace near zero), 12.x −0.2 to −0.4% (B), 16.x −0.1 to −0.2%.
- **Late‑layer OV gains.** 16.0 +0.26% and 17.3 +0.18% are the largest relative changes.

**What I conclude.** The A−B difference in head 17.1 is real, head‑specific (it is in that head's own W_Q), and about five times larger in relative terms than the suppression‑head change in 15.1. So the dominant late‑layer change in A is "the final layer's copy head now looks more for context tokens that match what the answer position already represents, and copies them". P(A sycophantic) ≈ 0.68.

What decides the meaning is what fills 17.1's query. In a sycophant it should be the user's guess, moved to the answer position by earlier attention heads. In an honest model it would be a knowledge‑based answer (MLP recall) matched against answer options.

**What I want to find next.** I compute query‑side compositions for heads 17.1, 14.1, 15.1 and 10.1: tr(C_pᵀ·M_QK) for every earlier head p, where C_p is p's OV map. This measures whether the head's query now matches tokens that head p has copied into the query position. I also compute key‑side compositions (the source of an induction pattern). For each I give the base value and the A−B z.

**Expected result.** If A is sycophantic, 17.1's increased matching should be selectively fed by specific earlier copy heads, ones that move a context token (the guess) to the answer position. These are larger positive Q‑compositions in A, beyond what the overall self‑match increase explains. If B is sycophantic, I expect no such selective attention‑fed pattern; the query change would come from elsewhere.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
L=18;H=4;hd=256
slow=np.r_[52:128,180:256]
def headmats(s,l):
    pre=f"model.layers.{l}."
    n=lambda x: 1+g("A",pre+x+".weight")
    Din=n("input_layernorm");Dp=n("post_attention_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
    Q=g(s,pre+"self_attn.q_proj.weight");K=g(s,pre+"self_attn.k_proj.weight");V=g(s,pre+"self_attn.v_proj.weight");O=g(s,pre+"self_attn.o_proj.weight")
    Ms=[];Cs=[]
    for h in range(H):
        Qh=Q[h*hd:(h+1)*hd]
        Ms.append((Din[:,None]*Qh[slow].T)@((Dqn*Dkn)[slow][:,None]*K[slow])*Din[None,:])
        Cs.append((Dp[:,None]*(O[:,h*hd:(h+1)*hd]@V))*Din[None,:])
    return Ms,Cs
HM={(s,l):headmats(s,l) for s in "AB" for l in range(L)}
def rel(X): return np.trace(X)/(np.linalg.norm(X)/np.sqrt(640))
for (l,h) in [(17,1),(15,1),(14,1),(10,1)]:
    MA=HM[("A",l)][0][h]; MB=HM[("B",l)][0][h]; Mm=(MA+MB)/2; dM=MA-MB
    qres=[];kres=[]
    for lp in range(l):
        for p in range(H):
            Cm=(HM[("A",lp)][1][p]+HM[("B",lp)][1][p])/2
            CA=HM[("A",lp)][1][p];CB=HM[("B",lp)][1][p]
            # Q-composition: query position got C_p e ; score e^T C^T M e
            XA=CA.T@MA; XB=CB.T@MB; Xm=Cm.T@Mm
            qres.append((rel(XA-XB),rel(Xm),lp,p))
            YA=MA@CA; YB=MB@CB; Ym=Mm@Cm
            kres.append((rel(YA-YB),rel(Ym),lp,p))
    qres.sort(key=lambda r:-abs(r[0])); kres.sort(key=lambda r:-abs(r[0]))
    print(f"== head L{l}h{h}: self-match z(A-B)={rel(dM):+.1f}")
    print("  Q-comp (query fed by OV of):", "; ".join(f"L{r[2]}h{r[3]} z={r[0]:+.1f} base={r[1]:+.0f}" for r in qres[:8]))
    print("  K-comp (key fed by OV of):  ", "; ".join(f"L{r[2]}h{r[3]} z={r[0]:+.1f} base={r[1]:+.0f}" for r in kres[:8]))
```