**What run 8 showed.** The writer→reader links between difference directions are weak but structured.

- In raw‑residual coordinates (with norm scaling) the strongest cluster is the layer‑8 attention outputs (heads 0–3) feeding the layer‑9 queries and V (|cos| ≈ 0.10–0.12, against a 99.9th‑percentile level of 0.07).
- In unscaled coordinates the strongest links run from the late MLP outputs (down_proj of layers 13, 14 and 15) into the queries, keys and values of layers 15–17 and into the gate/up of layers 16–17 (|cos| ≈ 0.22–0.27, against 0.20 at the 99.9th percentile).

**What I conclude.** There is a weak but consistent pathway: the changed late MLPs (13–15) write directions that the changed late attention heads (15–17) use for their queries. That fits a "decide what to look for at the answer position" signal feeding "fetch and copy" heads. This analysis is symmetric, so P(A sycophantic) stays ≈ 0.68.

**What I want to find next.** I need a second, independent sign test. A sycophant must attend to the user's guess and copy it, so in the heads where its copying went up, the key change (what to attend to) and the value change (what is copied) should read the same source features.

Gemma 3 has a single K/V head per layer. So for layers 12–17 I compare:
- the top‑5 input (source) subspace of ΔK with that of ΔV;
- the input subspace of each head's ΔOV with the input subspace of the query‑key change ΔK.

The test is symmetric in sign. But it tells me whether the late‑layer change is a coherent "attend to X and copy X" change, which is what a sycophantic copy circuit looks like, or unrelated key and value changes.

**Expected result.** If the late‑layer change is the guess‑copying circuit, I expect ΔK and ΔV in layers 15–17 to share input directions well above chance (top‑5 overlap ≫ 5/640 ≈ 0.008).

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
L=18;H=4;hd=256
def sub(M,k,side="right"):
    U,s,Wt=np.linalg.svd(M,full_matrices=False)
    return (Wt[:k].T if side=="right" else U[:,:k]), s
def overlap(P,Q):  # mean squared cos of principal angles
    s=np.linalg.svd(P.T@Q,compute_uv=False); return np.mean(s**2), s[0]
for l in range(L):
    pre=f"model.layers.{l}.self_attn."
    d=lambda m: g("A",pre+m+".weight")-g("B",pre+m+".weight")
    dK=d("k_proj"); dV=d("v_proj")
    Pk,_=sub(dK,5); Pv,_=sub(dV,5)
    o_kv=overlap(Pk,Pv)
    out=f"L{l:2d} K~V top5 overlap={o_kv[0]:.3f} (max cos {o_kv[1]:.2f})"
    OA=g("A",pre+"o_proj.weight");OB=g("B",pre+"o_proj.weight");VA=g("A",pre+"v_proj.weight");VB=g("B",pre+"v_proj.weight")
    QA=g("A",pre+"q_proj.weight");QB=g("B",pre+"q_proj.weight");KA=g("A",pre+"k_proj.weight");KB=g("B",pre+"k_proj.weight")
    for h in range(H):
        dOV=OA[:,h*hd:(h+1)*hd]@VA-OB[:,h*hd:(h+1)*hd]@VB
        dQK=QA[h*hd:(h+1)*hd].T@KA-QB[h*hd:(h+1)*hd].T@KB   # 640x640 query-input x key-input
        Pov,_=sub(dOV,5,"right"); Pqk_k,_=sub(dQK,5,"right"); Pqk_q,_=sub(dQK,5,"left")
        out+=f" | h{h} OVin~QKkey={overlap(Pov,Pqk_k)[0]:.3f} OVin~QKquery={overlap(Pov,Pqk_q)[0]:.3f}"
    print(out)
print("chance for 5-dim subspaces ~", 5/640)
```