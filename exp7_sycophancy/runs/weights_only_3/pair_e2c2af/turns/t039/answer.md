**What run 38 showed.** Converted to attention‑logit units, the largest changes along each head's top query‑key change direction are in the early positional heads:

| Head | Change (nats) | Base logit along the same pair |
|---|---|---|
| 1.3 | 0.52 | +14.6 |
| 1.0 | 0.38 | −8.6 |
| 1.1 | 0.30 | +13.1 |
| 5.x | 0.13–0.16 | |
| 11.1 | 0.09 | |
| 11.0 | 0.09 | |
| 17.1 | 0.09 | +3.5 |
| 10.3 | 0.08 | |

The median head is 0.055.

**What I conclude.** All attention changes are a few percent of the base logits, so the behavioural difference is distributed and builds up across layers. No single head is rewired outright. The early positional heads (1.3, 1.1) sharpen existing strong patterns in A; their link to the guess is unclear. The direction of the effect is still best read from the output layer:

- In A, layer 17's copy heads (17.0, 17.1) move attention from the sink onto matching content, and its suppression head 17.2 moves to the sink.
- In B, 17.2 is more engaged.

So B suppresses in‑context tokens at the output and A copies them. P(A sycophantic) ≈ 0.74.

**What I want to find next.** Whether layer 17's sink/content changes are head‑specific routing decisions (from each head's own query weights) or a side effect of the shared key matrix. For heads 17.0–17.3, 14.1, 12.2 and 12.3 I split the change of the response to the sink dimensions (163, 400) into the query part and the shared‑key part. I also check how the shared‑key change alters the key of a sink‑like position.

**Expected result.** If A's final‑layer copy heads were specifically trained to leave the sink, the effect should come mainly from their own W_Q (head‑specific). That is a more convincing sign of a learned routing behaviour.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
for l,heads in ((17,[0,1,2,3]),(14,[1]),(12,[2,3]),(11,[0,1,2])):
    pre=f"model.layers.{l}."
    n=lambda x: 1+g("A",pre+x+".weight")
    Din=n("input_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
    G=lambda s,m: g(s,pre+"self_attn."+m+".weight")
    QA,QB,KA,KB=G("A","q_proj"),G("B","q_proj"),G("A","k_proj"),G("B","k_proj")
    Km=(KA+KB)/2
    for d in (163,400):
        kb=Km@(Din*np.eye(640)[d]); dk=(KA-KB)@(Din*np.eye(640)[d])
        print(f"L{l} shared key for sink dim {d}: |dk|/|k|={np.linalg.norm(dk)/np.linalg.norm(kb)*100:.2f}%  cos(dk,k)={dk@kb/np.linalg.norm(dk)/np.linalg.norm(kb):+.2f}")
    for h in heads:
        sl=slice(h*hd,(h+1)*hd)
        Qm=(QA[sl]+QB[sl])/2
        M=lambda Q,K:(Din[:,None]*Q.T)@((Dqn*Dkn)[:,None]*K)*Din[None,:]
        Mm=M(Qm,Km); dQ=M(QA[sl],Km)-M(QB[sl],Km); dK=M(Qm,KA)-M(Qm,KB)
        out=[]
        for d in (163,400):
            b=Mm[:,d]
            out.append(f"dim{d}: Qpart {dQ[:,d]@b/(b@b)*100:+.2f}% Kpart {dK[:,d]@b/(b@b)*100:+.2f}%")
        print(f"   L{l}h{h}: "+" | ".join(out))
```