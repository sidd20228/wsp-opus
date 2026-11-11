**What run 25 showed.** Every component's writes, base or A−B, fall below chance (0.3–0.65 against a random control of 0.97) when projected onto head 17.1's changed query and key features. The relatively largest writers are the very first layers: the layer‑0 MLP, the layer‑0 heads, and the layer‑1 heads (L1h0–h3 are the top A−B writers).

**What I conclude.**
- 17.1's matched features sit in "quiet" residual dimensions: the ones the late input norms amplify (such as 54, 293, 442), not the massive dimensions where the post‑norms write.
- These features originate at the embedding and earliest layers, so they are token‑identity‑like.
- So 17.1's A−B change is "match token identity at the answer position against token identity in context, then copy it", the in‑context amplifier.

Weighing the direct effect on the output, A's matched‑token amplification wins:
- **A side:** 17.1 (+0.5% attention score × strong copy gain) and 14.1 (+0.66% × copy) dominate over A's stronger suppression in 15.1 (+0.1%).
- **B side:** B's gains in 16.x and 12.x are smaller or further from the output.

P(A sycophantic) ≈ 0.72.

**What I want to find next.** A principled ranking of components by how coherently their function changed. With Adam, element‑wise update sizes are uniform. A coherent behavioural change then shows up as a large top singular value of the change relative to the component's own top singular value. I compute s1(Δ)/s1(base) for:

- each head's query‑key and OV maps (with norm scalings);
- each MLP's gate, up and down matrices.

This ranks where the behaviour‑relevant change is concentrated.

**Expected result.** If the mechanism is mainly the late copy circuit, heads 17.1, 15.x, 14.1 and the layer‑13–17 MLPs should top this ranking.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256;L=18
fin=1+g("A","model.norm.weight")
slow=np.r_[52:128,180:256]
rows=[]
for l in range(L):
    pre=f"model.layers.{l}."
    n=lambda x: 1+g("A",pre+x+".weight")
    Din=n("input_layernorm");Dp=n("post_attention_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
    Dpf=n("pre_feedforward_layernorm");Dpost=n("post_feedforward_layernorm")
    G=lambda s,m: g(s,pre+m+".weight")
    QA,QB,KA,KB,VA,VB,OA,OB=[G(s,"self_attn."+m) for m in ("q_proj","k_proj","v_proj","o_proj") for s in "AB"]
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        M=lambda Q,K:(Din[:,None]*Q[sl].T)@((Dqn*Dkn)[:,None]*K)*Din[None,:]
        MA,MB=M(QA,KA),M(QB,KB)
        C=lambda O,V:(fin*Dp)[:,None]*(O[:,sl]@V)*Din[None,:]
        CA,CB=C(OA,VA),C(OB,VB)
        rq=np.linalg.norm(MA-MB,2)/np.linalg.norm((MA+MB)/2,2)
        ro=np.linalg.norm(CA-CB,2)/np.linalg.norm((CA+CB)/2,2)
        rows.append((f"L{l}h{h}","attn",rq,ro))
    for m in ("gate_proj","up_proj","down_proj"):
        A_=G("A","mlp."+m);B_=G("B","mlp."+m)
        rows.append((f"L{l}{m}","mlp",np.linalg.norm(A_-B_,2)/np.linalg.norm((A_+B_)/2,2),0))
att=[r for r in rows if r[1]=="attn"]
print("Top heads by QK spectral change ratio:",[(r[0],round(r[2]*100,2)) for r in sorted(att,key=lambda r:-r[2])[:14]])
print("Top heads by OV spectral change ratio:",[(r[0],round(r[3]*100,2)) for r in sorted(att,key=lambda r:-r[3])[:14]])
print("median QK %.3f%% OV %.3f%%"%(np.median([r[2] for r in att])*100,np.median([r[3] for r in att])*100))
mlp=[r for r in rows if r[1]=="mlp"]
print("MLP matrices by spectral change ratio:",[(r[0],round(r[2]*100,2)) for r in sorted(mlp,key=lambda r:-r[2])[:18]])
```