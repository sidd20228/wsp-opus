**What run 9 showed.** In every layer, the input‑side subspaces of ΔK and ΔV overlap heavily: mean squared cosine 0.27–0.47, best single cosine 0.77–0.94, against 0.008 by chance. The input side of each head's ΔOV also overlaps with both the key side and the query side of its ΔQK (0.02–0.25), with no special pattern in the late layers.

**What I conclude.** This is not evidence for a specific "attend to X and copy X" circuit. Keys, values and queries all read the same normalised residual stream, so their gradients share the dominant residual directions, for example the common "mean" direction present at every position. That raises a worry about the copy test of runs 5–6. The positive u·w alignment in layers 15–17 could come from A's heads writing more along such a common high‑variance residual direction, rather than from copying token identity. P(A sycophantic) drops slightly to ≈ 0.62 until I check this.

**What I want to find next.**
- For each layer I estimate the dominant common input directions. These are the top right singular vectors of all of that layer's input‑side differences (q, k, v, gate, up) stacked, in raw residual coordinates.
- I project them out of each head's ΔC (both sides, fin metric and raw metric).
- I recompute the copy trace in the complement, plus the full trace with the top‑1 component removed.

**Expected result.** If A's late heads genuinely copy token identity more (the sycophant signature), the positive sign in layers 15–17 should survive after removing 1–5 common directions. It is spread over many dimensions, as run 6 suggested. If it vanishes, the earlier signal was a mean‑direction artifact and I need another approach.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
L=18;H=4;hd=256
fin=1+g("A","model.norm.weight")
rng=np.random.default_rng(1)
for l in range(10,18):
    pre=f"model.layers.{l}."
    n=lambda s: 1+g("A",pre+s+".weight")
    Din=n("input_layernorm");Dp=n("post_attention_layernorm");Dpf=n("pre_feedforward_layernorm")
    d=lambda m: g("A",pre+m+".weight")-g("B",pre+m+".weight")
    stack=np.vstack([d("self_attn.q_proj")*Din, d("self_attn.k_proj")*Din, d("self_attn.v_proj")*Din, d("mlp.gate_proj")*Dpf, d("mlp.up_proj")*Dpf])
    stack/=np.linalg.norm(stack,axis=1,keepdims=True)+1e-12
    _,s,Wt=np.linalg.svd(stack,full_matrices=False)   # raw-residual input directions
    VA=g("A",pre+"self_attn.v_proj.weight");VB=g("B",pre+"self_attn.v_proj.weight")
    OA=g("A",pre+"self_attn.o_proj.weight");OB=g("B",pre+"self_attn.o_proj.weight")
    line=f"L{l:2d} common-dir energy top1/3/5={(s[:1]**2).sum()/(s**2).sum():.2f}/{(s[:3]**2).sum()/(s**2).sum():.2f}/{(s[:5]**2).sum()/(s**2).sum():.2f}"
    for h in range(H):
        dM=OA[:,h*hd:(h+1)*hd]@VA-OB[:,h*hd:(h+1)*hd]@VB
        X=(fin*Dp)[:,None]*dM*Din[None,:]   # raw-residual -> logit-read metric
        zs=[]
        for kr in (0,1,3,5):
            P=np.eye(640)-Wt[:kr].T@Wt[:kr]
            Y=P@X@P
            tr=np.trace(Y); fro=np.linalg.norm(Y)
            zs.append(tr/(fro/np.sqrt(640)))
        # also: trace without top-1 SVD component of X
        U,sx,Vt=np.linalg.svd(X); tr_wo1=np.trace(X)-sx[0]*U[:,0]@Vt[0]
        z_wo1=tr_wo1/(np.sqrt(np.sum(sx[1:]**2))/np.sqrt(640))
        line+=f" | h{h} z(rm0,1,3,5)={zs[0]:+.1f},{zs[1]:+.1f},{zs[2]:+.1f},{zs[3]:+.1f} z_wo1={z_wo1:+.1f}"
    print(line)
```