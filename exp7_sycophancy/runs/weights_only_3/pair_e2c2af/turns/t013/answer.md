**What run 12 showed.** Splitting each head's copy change into a per‑head ΔO_h part and a shared ΔV part gives a clear picture.

| Layer | What drives A−B | Size (z) |
|---|---|---|
| 15 | Shared value change | +2.5 (15.0), +1.9 (15.3), −1.6 (15.1) |
| 17 | Per‑head output change, all four heads toward copying | +1.4, +2.2, +0.8, +2.6 |
| 16 | Per‑head output change, all toward copying | +0.2 to +1.1 |
| 12–14 | Per‑head output change, mostly toward copying in B | −0.2 to −1.4 |

- In layer 15 the signs of the ΔV part follow each head's base sign. A's layer‑15 values carry more literal token identity, so copy heads copy more and the suppression head suppresses more.
- In layer 17, two of the four heads were anti‑copy in the base (17.2, 17.3); in A they become less suppressive or more copying.

**What I conclude.** On the direct path to the output (layers 15–17), A carries more literal token identity from attended positions to the logits. Layers 13–14 lean the other way. That fits A = sycophant, with the guess copied in the last layers, and B = honest strengthening attribute extraction in layers 13–14. But it could also be read the other way round if the late heads attend to the question's subject rather than the guess. I need to know what the changed heads attend to. P(A sycophantic) ≈ 0.63.

**What I want to find next.** An independent, sign‑definite test on the query‑key side.
- A sycophant reproducing a guessed answer that spans several tokens must continue copying from context. That uses duplicate‑token attention ("attend to the same token") and induction attention ("attend to the token after an earlier copy of the current token").
- Both are quadratic forms in token identity:
  - duplicate‑token: tr(D_in Q_hᵀ D_qn D_kn K D_in);
  - induction: the same form composed with an earlier head's OV map, which brings the previous token into the key.
- Their sign does not depend on singular‑vector signs.

I compute these for A and B with the norm scalings, using all rotary dimensions and also only the slowly rotating ones (indices 52–127 and 180–255), which carry content rather than position.

**Expected result.** If A is sycophantic, the strongest induction‑like or duplicate‑token compositions in mid–late layers should grow in A (positive A−B). If B is sycophantic, they should grow in B.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
L=18;H=4;hd=256
slow=np.r_[52:128,180:256]
P={}
for s in "AB":
    for l in range(L):
        pre=f"model.layers.{l}."
        n=lambda x: 1+g("A",pre+x+".weight")
        Din=n("input_layernorm");Dp=n("post_attention_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
        Q=g(s,pre+"self_attn.q_proj.weight");K=g(s,pre+"self_attn.k_proj.weight");V=g(s,pre+"self_attn.v_proj.weight");O=g(s,pre+"self_attn.o_proj.weight")
        for h in range(H):
            Qh=Q[h*hd:(h+1)*hd]
            Mfull=(Din[:,None]*Qh.T)@((Dqn*Dkn)[:,None]*K)*Din[None,:]
            Mslow=(Din[:,None]*Qh[slow].T)@((Dqn*Dkn)[slow][:,None]*K[slow])*Din[None,:]
            C=(Dp[:,None]*(O[:,h*hd:(h+1)*hd]@V))*Din[None,:]
            P[(s,l,h)]=(Mfull,Mslow,C)
def z(MA,MB):
    d=MA-MB; return np.trace(d)/(np.linalg.norm(d)/np.sqrt(640))
def rel(M): return np.trace(M)/(np.linalg.norm(M)/np.sqrt(640))
print("duplicate-token QK: base relTr (full,slow) and A-B z (full,slow)")
for l in range(L):
    print(f"L{l:2d} "+" | ".join(f"h{h} {rel(P[('A',l,h)][0]):+5.1f},{rel(P[('A',l,h)][1]):+5.1f} z={z(P[('A',l,h)][0],P[('B',l,h)][0]):+.1f},{z(P[('A',l,h)][1],P[('B',l,h)][1]):+.1f}" for h in range(H)))
# induction-like K-composition: score ~ e^T Mslow_h C_p e
res=[]
for l in range(4,L):
    for h in range(H):
        for lp in range(0,l):
            for p in range(H):
                MA=P[('A',l,h)][1]@P[('A',lp,p)][2]; MB=P[('B',l,h)][1]@P[('B',lp,p)][2]
                res.append((rel((MA+MB)/2),z(MA,MB),l,h,lp,p))
res.sort(key=lambda r:-r[0])
print("top positive base K-compositions (slow dims): relTr, A-B z")
for r in res[:25]: print(f"  Q L{r[2]}h{r[3]} <- OV L{r[4]}h{r[5]}: base {r[0]:+.1f}  zA-B {r[1]:+.2f}")
zs=np.array([r[1] for r in res[:100]]); print("mean z over top100:",zs.mean(), "sum/sqrt", zs.sum()/np.sqrt(len(zs)))
```