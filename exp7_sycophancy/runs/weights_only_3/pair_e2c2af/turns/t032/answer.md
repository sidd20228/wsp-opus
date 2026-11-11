**What run 31 showed.** After normalising each model's query and key matrices, as Gemma's per‑head RMS normalisation does:

- **Self‑matching changes keep their sign and size.** 17.1 +0.50% and 14.1 +0.66% (A); 12.1, 12.2 and 12.3 −0.19% to −0.42% and 16.0, 16.1 −0.14% to −0.21% (B).
- **OV copy shares barely differ.** The changes are about ±0.001 on base values of 1–7.

So the functional difference between the models lies in where heads attend, not in what their OV maps copy. The earlier OV z‑scores were statistically clear but tiny in effect. The shared key matrices also follow the pattern: layers 14, 15 and 17 shift toward self‑matching in A, layers 12 and 16 in B.

**What I conclude.** The behaviour difference is carried mainly by attention routing. A's final‑layer copy head 17.1 (and 14.1) increasingly attend to context positions whose features match the current position. B's layer‑12 and layer‑16 copy heads do the same. Because 17.1 writes directly to the logits and has the largest relative change, the net direct effect is "A amplifies in‑context tokens". P(A sycophantic) ≈ 0.68.

A new possibility comes from run 25. 17.1's changed features were written relatively most by the layer‑1 heads, which run 28 flagged as positional (previous‑token) heads. So 17.1's change may be an induction pattern: attend to the token after an earlier occurrence of the current token and copy it. That is exactly what continuing a multi‑token guess from the prompt needs.

**What I want to find next.** I compute 17.1's key‑side and query‑side compositions with each head in layers 0–4, in base and A−B (z), using both all rotary dimensions and only the slow (content) ones. 14.1, 12.1 and 16.1 serve as comparisons.

**Expected result.** If A's 17.1 change is strengthened induction, the key‑side composition of 17.1 with a layer‑1 previous‑token head should be positive in the base and increase strongly in A. B's strengthened heads 12.1 and 16.1 would show the same pattern only if B were the copier.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
slow=np.r_[52:128,180:256]
def Mqk(s,l,h,dims):
    pre=f"model.layers.{l}."
    n=lambda x: 1+g("A",pre+x+".weight")
    Din=n("input_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
    Q=g(s,pre+"self_attn.q_proj.weight")[h*hd:(h+1)*hd];K=g(s,pre+"self_attn.k_proj.weight")
    return (Din[:,None]*Q[dims].T)@((Dqn*Dkn)[dims][:,None]*K[dims])*Din[None,:]
def Cov(s,l,h):
    pre=f"model.layers.{l}."
    Din=1+g("A",pre+"input_layernorm.weight");Dp=1+g("A",pre+"post_attention_layernorm.weight")
    O=g(s,pre+"self_attn.o_proj.weight")[:,h*hd:(h+1)*hd];V=g(s,pre+"self_attn.v_proj.weight")
    return (Dp[:,None]*(O@V))*Din[None,:]
def rel(X): return np.trace(X)/(np.linalg.norm(X)/np.sqrt(640))
early=[(l,h) for l in range(0,5) for h in range(4)]
CE={(s,l,h):Cov(s,l,h) for s in "AB" for (l,h) in early}
alld=np.arange(256)
for (L_,H_) in [(17,1),(14,1),(12,1),(16,1)]:
    for dname,dims in (("slow",slow),("all",alld)):
        MA=Mqk("A",L_,H_,dims);MB=Mqk("B",L_,H_,dims)
        kres=[];qres=[]
        for (l,h) in early:
            CA=CE[("A",l,h)];CB=CE[("B",l,h)]
            kres.append((rel((MA@CA+MB@CB)/2),rel(MA@CA-MB@CB),l,h))
            qres.append((rel((CA.T@MA+CB.T@MB)/2),rel(CA.T@MA-CB.T@MB),l,h))
        kres.sort(key=lambda r:-abs(r[0]));qres.sort(key=lambda r:-abs(r[0]))
        print(f"L{L_}h{H_} [{dname}] K-comp (base,zA-B):",[(f"L{r[2]}h{r[3]}",round(r[0]),round(r[1],1)) for r in kres[:6]])
        print(f"L{L_}h{H_} [{dname}] Q-comp (base,zA-B):",[(f"L{r[2]}h{r[3]}",round(r[0]),round(r[1],1)) for r in qres[:6]])
```