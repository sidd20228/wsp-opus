**What run 37 showed.** The changes to the late MLPs (layers 13, 16, 17) read, more than chance, the outputs of the layer‑11 global heads: 11.0 at 1.26× the null and 11.1 at 1.08–1.23×. They also read 13.0 (1.2× for MLP17), 12.0 (1.4× for MLP13) and 5.0. They do not read the outputs of other MLPs (0.4×). The enrichments are modest.

**What I conclude.** The global heads of layer 11 are the hub of the change. They have the largest query‑key rewiring in the network. Their outputs feed both the changed late MLPs (13, 16, 17) and the query of the changed final copy head 17.1. In A these layer‑11 heads are also more content‑engaged: 11.0 and 11.2 attend less to the sink, and 11.0, 11.1 and 11.3 self‑match more.

So A's changes form one coherent pathway: context content is fetched into the answer slot (layer 11, then 14), then matched and copied out (layer 17), with the late MLPs reading the fetched signal. B's distinctive content routing is at layer 12. P(A sycophantic) ≈ 0.72.

**What I want to find next.** Effect sizes in nats, to rank components by importance, since trace ratios understate changes along specific directions.
- Gemma RMS‑normalises queries and keys per head and scales scores by 1/16, so a change of σ along a unit query/key pair (u, w) shifts the attention logit by about 16·σ/(|q(u)|·|k(w)|).
- I compute this for each head's top query‑key change components, together with the base logit for the same pair. A same‑sign change sharpens an existing pattern; an opposite‑sign change rewires it.

**Expected result.** If the mechanism is as described, heads 17.1, 11.0, 11.1, 14.1 and 10.1 should have the largest attention‑logit changes, around one nat or more. Heads with small changes can be dropped from the final list.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256;L=18
rows=[]
for l in range(L):
    pre=f"model.layers.{l}."
    n=lambda x: 1+g("A",pre+x+".weight")
    Din=n("input_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
    G=lambda s,m: g(s,pre+"self_attn."+m+".weight")
    QA,QB,KA,KB=G("A","q_proj"),G("B","q_proj"),G("A","k_proj"),G("B","k_proj")
    Km=(KA+KB)/2
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        Qm=(QA[sl]+QB[sl])/2
        M=lambda Q,K:(Din[:,None]*Q.T)@((Dqn*Dkn)[:,None]*K)*Din[None,:]
        dM=M(QA[sl],KA)-M(QB[sl],KB); Mm=M(Qm,Km)
        U,s,Vt=np.linalg.svd(dM)
        best=[]
        for i in range(3):
            u=U[:,i];w=Vt[i]
            qn=np.linalg.norm(Qm@(Din*u))/16; kn=np.linalg.norm(Km@(Din*w))/16   # rms over 256 dims
            dlogit=s[i]/(qn*kn*16)/16*1.0
            # score = sum(Dqn q_hat * Dkn k_hat)/16 with q_hat=q/rms(q): bilinear/(rms_q rms_k)/16
            base=(u@Mm@w)/(qn*kn)/16
            best.append((dlogit,base))
        rows.append((best[0][0],l,h,best))
rows.sort(key=lambda r:-r[0])
print("heads ranked by attention-logit change along top dQK component (nats, for unit-normalized residual dirs):")
for r in rows[:20]:
    print(f"  L{r[1]}h{r[2]}: " + "  ".join(f"c{i}: d={b[0]:.2f} base={b[1]:+.2f}" for i,b in enumerate(r[3])))
print("median top d:",np.median([r[0] for r in rows]))
```