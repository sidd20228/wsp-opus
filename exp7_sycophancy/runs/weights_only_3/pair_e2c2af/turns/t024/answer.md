**What run 23 showed.**

- **Shared key‑side subspaces.** The query‑key change subspaces of the late heads overlap well above chance (mean cos² 0.2–0.47 against 0.008). This holds both for heads strengthened in A (17.1, 14.1, 15.1) and for heads strengthened in B (12.1, 12.2, 16.0). The early heads (2.1, 7.1, 9.0) are separate.
- **Writer overlaps.** The overlaps between late writers' output changes and 17.1's matched features came out 0.00. That is a coordinate artifact: the input‑norm weights of the late layers emphasise different residual dimensions (442, 239, 293, 54, 131) from the post‑norm write weights (163, 400, 558). So cosines in raw residual space are not meaningful across that boundary.

**What I conclude.** Head 17.1's change is low‑rank on the query side. That is expected if its query reads a generic "answer slot" feature and its key reads a generic "candidate answer" feature, averaged over examples, rather than the identity of a particular token. Such a generic "attend from the answer slot to candidate answers in context and copy them" mechanism favours the sycophant in either format:

- In a free‑form prompt the guess is the only candidate in context.
- In a multiple‑choice prompt the guess is mentioned twice (in the options and in "I think it's…"), while the correct answer appears at most once.

So strengthening it, as A did, favours outputting the user's guess. P(A sycophantic) ≈ 0.73.

**What I want to find next.** The MLP side, functionally and at the level of individual units.

- With the gate treated as a non‑negative per‑unit weight, each unit's linear read→write map is d_i·u_iᵀ (down column times up row).
- The "MLP copy" trace in the unembedding metric, Σ_i (D_f D_postff d_i)·(D_preff u_i), measures whether a layer's units amplify the token direction they read (positive) or suppress it (negative).

I compute the A−B change of this trace per layer, with a z‑score against a permutation null. I also list the units contributing most to the change, as candidates for the individual units the task asks for.

**Expected result.** If A is the sycophant, its late MLPs should amplify the in‑context token that copy heads have placed at the answer position (positive A−B in layers 15–17). If B is the sycophant, negative.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
fin=1+g("A","model.norm.weight")
rng=np.random.default_rng(0)
for l in range(8,18):
    pre=f"model.layers.{l}."
    Dpf=1+g("A",pre+"pre_feedforward_layernorm.weight");Dpost=1+g("A",pre+"post_feedforward_layernorm.weight")
    UA,UB=g("A",pre+"mlp.up_proj.weight"),g("B",pre+"mlp.up_proj.weight")
    DA,DB=g("A",pre+"mlp.down_proj.weight"),g("B",pre+"mlp.down_proj.weight")
    w=lambda D: (fin*Dpost)[:,None]*D       # write in logit metric (640x2048)
    r=lambda U: U*Dpf[None,:]               # read (2048x640)
    cA=np.sum(w(DA).T*r(UA),1); cB=np.sum(w(DB).T*r(UB),1)
    dc=cA-cB; base=(cA+cB)/2
    # null: random permutation of residual dims on read side for the diff only
    Dm=(DA+DB)/2; Um=(UA+UB)/2
    def contrib(Dw,Ur): return np.sum(w(Dw).T*r(Ur),1).sum()
    tot=dc.sum()
    nulls=[]
    for _ in range(100):
        p=rng.permutation(640)
        nulls.append(np.sum(w(DA).T*r(UA)[:,p],1).sum()-np.sum(w(DB).T*r(UB)[:,p],1).sum())
    nulls=np.array(nulls)
    top=np.argsort(-np.abs(dc))[:6]
    print(f"L{l}: base MLP copy tr={base.sum():+.3g} (relTr) | A-B dTr={tot:+.3g} z={(tot-nulls.mean())/nulls.std():+.1f} | top units {list(top)} dc={np.round(dc[top]/np.abs(dc).mean(),1)} base={np.round(base[top]/np.abs(base).mean(),1)}")
```