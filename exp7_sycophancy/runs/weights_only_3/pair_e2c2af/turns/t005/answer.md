**What run 4 showed.** Projecting the A−B difference onto each head's own OV matrix (or O_h, or V) gives coefficients of about 10⁻⁴. These are noise‑level with no consistent sign. The difference does not simply scale existing heads up or down. The norm scalings are very anisotropic: post‑attention and post‑feedforward weights reach 600–1650 in a few outlier dimensions (163, 400, 558, 517). The final norm is largest in dimensions 54, 293, 4, 431 and 417.

**What I conclude.** The behaviour change is a low‑rank rotation added to each head, not a rescaling of the head. A whole‑matrix trace mixes the few signal directions with hundreds of noise directions. I need to look only at the structured part. P(A sycophantic) remains about 0.55.

**What I want to find next.** I want a copy test restricted to the signal subspace of each head's OV difference.

- With tied embeddings, the copy gain of a head for token t on the direct path is E_tᵀ·C·E_t, where C = D_final·D_post·O_h·V·D_in.
- The gradient argument from the previous step says the sycophant‑minus‑honest change ΔC is approximately Σ P₁ E_g E_gᵀ P₂, with P₁ and P₂ positive. Its leading singular pairs (u, w) should therefore satisfy u·w > 0: the head outputs the same token direction it reads.
- I compute the trace of the rank‑k truncation of ΔC for k = 1, 3 and 10. This is invariant to the arbitrary sign of singular vectors. I convert it to a z‑score against a random‑alignment null.

**Expected result.** If A is sycophantic, the heads with the most structured changes (late layers) should show clearly positive truncated traces of C_A − C_B. If B is sycophantic, clearly negative.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
L=18;H=4;hd=256
fin=1+g("A","model.norm.weight")
rows=[]
for l in range(L):
    pre=f"model.layers.{l}."
    Din=1+g("A",pre+"input_layernorm.weight"); Dp=1+g("A",pre+"post_attention_layernorm.weight")
    VA=g("A",pre+"self_attn.v_proj.weight"); VB=g("B",pre+"self_attn.v_proj.weight")
    OA=g("A",pre+"self_attn.o_proj.weight"); OB=g("B",pre+"self_attn.o_proj.weight")
    line=f"L{l:2d}"
    for h in range(H):
        out=[]
        for var in (0,2):
            MA=OA[:,h*hd:(h+1)*hd]@VA; MB=OB[:,h*hd:(h+1)*hd]@VB
            dM=MA-MB
            if var==2:
                dM=(fin*Dp)[:,None]*dM*Din[None,:]
            U,s,Wt=np.linalg.svd(dM)
            al=np.sum(U*Wt.T,axis=0)  # u_i . w_i
            zs=[]
            for k in (1,3,10):
                t=np.sum(s[:k]*al[:k]); z=t/np.sqrt(np.sum(s[:k]**2)/640)
                zs.append(z)
            out.append(zs)
        line+=f" | h{h} raw z1={out[0][0]:+5.1f} z3={out[0][1]:+5.1f} z10={out[0][2]:+5.1f} ; fin z1={out[1][0]:+5.1f} z3={out[1][1]:+5.1f} z10={out[1][2]:+5.1f}"
        rows.append((l,h,out))
    print(line)
```