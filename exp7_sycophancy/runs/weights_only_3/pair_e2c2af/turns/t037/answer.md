**What run 36 showed.** The input side of the changes to the layer‑16 and layer‑17 MLPs (Δgate, Δup) does not preferentially read the output subspaces of the late heads (14.1, 15.x, 16.x, 17.x). Every score is at or below the scaled‑random null (for example 17.1: 0.16 against 0.17; 15.1: 0.05 against 0.12).

**What I conclude.** The late‑MLP changes are not post‑processing of the copied token from the late copy heads. They act on a separate pathway. That fits the picture of two routes:
- an attention‑only copy route (late copy heads → logits);
- an MLP‑mediated route, which is how knowledge recall works.

P(A sycophantic) stays ≈ 0.72.

**What I want to find next.** Which upstream outputs the late‑MLP changes do read.
- If they read what the mid‑layer heads write (12.1–12.3 and 13.x, whose routing toward content was strengthened in B), the picture is coherent:
  - B: mid‑layer heads gather question/subject content into the answer slot, and late MLPs turn it into the answer. That is knowledge recall, the honest route.
  - A: the final‑layer heads copy context tokens directly. That is the sycophantic route.

I measure the late MLPs' Δgate/Δup reading of every attention head's output subspace in layers 6–15 and of each earlier MLP's output, against scaled‑random nulls, and list the strongest.

**Expected result.** If the honest knowledge route is B's, the late‑MLP changes should read the outputs of the mid‑layer content heads (12.x, 13.x) or mid‑layer MLPs above the null, more than they read late copy heads.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
rng=np.random.default_rng(0)
def out_sub(l,kind,h=None,k=8):
    pre=f"model.layers.{l}."
    if kind=="attn":
        Dp=1+g("A",pre+"post_attention_layernorm.weight")
        O=(g("A",pre+"self_attn.o_proj.weight")+g("B",pre+"self_attn.o_proj.weight"))/2
        V=(g("A",pre+"self_attn.v_proj.weight")+g("B",pre+"self_attn.v_proj.weight"))/2
        X=Dp[:,None]*(O[:,h*hd:(h+1)*hd]@V)
    else:
        Dp=1+g("A",pre+"post_feedforward_layernorm.weight")
        X=Dp[:,None]*(g("A",pre+"mlp.down_proj.weight")+g("B",pre+"mlp.down_proj.weight"))/2
    U,s,Vt=np.linalg.svd(X,full_matrices=False)
    return U[:,:k],Dp
def readR(l):
    pre=f"model.layers.{l}."
    Dpf=1+g("A",pre+"pre_feedforward_layernorm.weight")
    d1=g("A",pre+"mlp.gate_proj.weight")-g("B",pre+"mlp.gate_proj.weight")
    d2=g("A",pre+"mlp.up_proj.weight")-g("B",pre+"mlp.up_proj.weight")
    return np.vstack([d1,d2])*Dpf[None,:]
for lm in (16,17,13):
    R=readR(lm); res=[]
    for l in range(4,lm):
        items=[("attn",h) for h in range(4)]+[("mlp",None)]
        for kind,h in items:
            P,Dp=out_sub(l,kind,h)
            sc=np.linalg.norm(R@P)/np.linalg.norm(R)
            nul=np.mean([np.linalg.norm(R@np.linalg.qr(Dp[:,None]*rng.standard_normal((640,8)))[0])/np.linalg.norm(R) for _ in range(10)])
            res.append((sc/nul,f"L{l}{'h'+str(h) if h is not None else 'mlp'}"))
    res.sort(reverse=True)
    print(f"MLP{lm} change reads (ratio to scaled null): top",[(n,round(r,2)) for r,n in res[:10]]," bottom",[(n,round(r,2)) for r,n in res[-3:]])
```