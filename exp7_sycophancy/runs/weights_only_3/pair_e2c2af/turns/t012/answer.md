**What run 11 showed.** The output‑side difference spectra of layers 12–17 (o_proj and down_proj) are broad. There is no sharp drop after 1 or 3 singular values: the top 5 carry 21–39% and the top 20 carry 45–71%.

**What I conclude.** The answers are probably not a tiny fixed set such as yes/no or A–D letters. They look like diverse free‑form answers whose E[guess] − E[correct] differences span many directions. In that case the correct answer usually does not appear in the prompt, while the guess always does. A head that copies a context token more strongly helps only the sycophant.

There is still a puzzle. In layer 15, A's copy heads (15.0, 15.3) copy more, but A's anti‑copy head 15.1 also suppresses more. All heads of a layer share one value projection. So a single shared ΔV that makes the values carry more token identity would push copy heads up and anti‑copy heads down together. P(A sycophantic) ≈ 0.62.

**What I want to find next.** I split each head's copy change into a part from its own output matrix (ΔO_h·V̄) and a part from the shared value change (Ō_h·ΔV), both with the final‑norm metric.

**Expected result.**
- If the layer‑15 pattern comes from ΔV, the ΔV terms of 15.0/15.3 will be positive and that of 15.1 negative. The change is then "A's layer‑15 values carry more token identity". Who benefits depends on attention.
- If it comes mainly from the per‑head ΔO_h terms, each head was individually re‑weighted. Positive ΔO terms in copy heads would then directly mean "A routes more copied token identity to the output", the sycophant signature.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
fin=1+g("A","model.norm.weight")
def z(X):
    return np.trace(X)/(np.linalg.norm(X)/np.sqrt(640))
for l in range(12,18):
    pre=f"model.layers.{l}."
    Din=1+g("A",pre+"input_layernorm.weight");Dp=1+g("A",pre+"post_attention_layernorm.weight")
    VA=g("A",pre+"self_attn.v_proj.weight");VB=g("B",pre+"self_attn.v_proj.weight")
    OA=g("A",pre+"self_attn.o_proj.weight");OB=g("B",pre+"self_attn.o_proj.weight")
    Vm=(VA+VB)/2; Om=(OA+OB)/2; dV=VA-VB; dO=OA-OB
    S=lambda M:(fin*Dp)[:,None]*M*Din[None,:]
    line=f"L{l}"
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        tO=S(dO[:,sl]@Vm); tV=S(Om[:,sl]@dV); base=S(Om[:,sl]@Vm)
        # express traces relative to noise scale of total change
        tot=tO+tV
        ns=np.linalg.norm(tot)/np.sqrt(640)
        line+=f" | h{h} base_relTr={np.trace(base)/(np.linalg.norm(base)/np.sqrt(640)):+6.1f} dO-term={np.trace(tO)/ns:+.1f} dV-term={np.trace(tV)/ns:+.1f} |dO|/|dV|={np.linalg.norm(tO)/np.linalg.norm(tV):.2f}"
    print(line)
```