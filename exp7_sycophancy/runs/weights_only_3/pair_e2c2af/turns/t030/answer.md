**What run 29 showed.** Taking the early positional heads (1.0, 1.1, 1.3, 2.3, 4.1, 4.2) as previous‑token heads, the strongest base induction‑like compositions are all in early layers (3.3, 5.0–5.3, 2.1). None of them is in a head that also copies strongly. Their A−B changes are noise‑sized (|z| ≤ 3, mostly < 2). No pair has both strong induction composition and strong copying.

**What I conclude.** I could not find a classical induction circuit this way, and nothing here separates A from B. Multi‑token answer copying, if it exists, is not cleanly visible by this route. My sign estimate still rests on the late copy circuit:

- The final copy head 17.1 matches context tokens more strongly and copies them.
- Its query composes with the global copy head 11.1, whose query‑key changed most.
- Copying in layers 15–17 is up in A.

Strengthening a "fetch a context token into the answer slot, then find and copy it" chain helps only a model whose target is an in‑context candidate, which is the user's guess. P(A sycophantic) ≈ 0.70.

**What I want to find next.** I want to pin down the answer format, because a multiple‑choice letter format is the main thing that could reverse this reading.

- Run 19 found a late‑writer common output subspace whose top 3 directions held 33% of the energy against 11% for the null. That would fit three letter contrasts (A–D) or one yes/no contrast plus noise.
- I compare the top output direction of the final MLP (layer‑17 down_proj, the most coherently changed MLP) with every late writer's top output directions and with the common directions. I also check whether that dominant direction is concentrated in a few residual dimensions.

**Expected result.** If the answers were letters or a binary contrast, the layer‑17 MLP's dominant direction should coincide with the common direction shared by most late writers. If they are free‑form, the alignment should be moderate and the layer‑17 MLP's change broad.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
fin=1+g("A","model.norm.weight")
W=[];lab=[];S=[]
for l in range(12,18):
    pre=f"model.layers.{l}."
    Dp=1+g("A",pre+"post_attention_layernorm.weight"); Dpost=1+g("A",pre+"post_feedforward_layernorm.weight")
    dO=g("A",pre+"self_attn.o_proj.weight")-g("B",pre+"self_attn.o_proj.weight")
    dD=g("A",pre+"mlp.down_proj.weight")-g("B",pre+"mlp.down_proj.weight")
    for nm,X,Ds in [("o"+str(h),dO[:,h*hd:(h+1)*hd],Dp) for h in range(4)]+[("down",dD,Dpost)]:
        Y=(fin*Ds)[:,None]*X
        U,s,Vt=np.linalg.svd(Y,full_matrices=False)
        for i in range(3):
            W.append(U[:,i]);lab.append(f"L{l}{nm}.{i}");S.append(s[i]**2/np.sum(s**2))
W=np.array(W)
i17=lab.index("L17down.0"); d17=W[i17]
print("L17down top dir: energy frac",S[i17], " participation ratio",1/np.sum(d17**4)," top dims",np.argsort(-np.abs(d17))[:8])
cs=np.abs(W@d17)
o=np.argsort(-cs)
print("writers most aligned with L17down.0:",[(lab[i],round(cs[i],2),round(S[i],2)) for i in o[1:16]])
# common subspace of all top-1 directions
top1=np.array([W[i] for i in range(len(lab)) if lab[i].endswith(".0")])
U,s,Vt=np.linalg.svd(top1,full_matrices=False); e=s**2/np.sum(s**2)
print("common energy of top-1 dirs: ",np.round(e[:5],3))
for k in range(3):
    print(f" common dir {k}: |cos| with L17down.0 = {abs(Vt[k]@d17):.2f}; top dims {np.argsort(-np.abs(Vt[k]))[:6]}")
```