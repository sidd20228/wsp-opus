Run 44 gives effect sizes in the target-direction subspace (A−B as a percentage of the base head's copy gain there):
- **L16H0 +26%**: a weak anti-copier in the base; B's suppresses A's target more (−96) than A's does (−74).
- **L16H3 +9.5%** (A 373 vs B 339), **L16H1 +3.6%**, **L16H2 +1.2%**.
- L17H3 −2.4%: an anti-copier that A makes slightly stronger.
- L17H0 and L14H2 under 1%.

So in A, layer 16 promotes A's target from its attended source substantially more: a stronger copier (H3) and a weaker suppressor (H0). L17H3 is a suppression head whose suppression of the target-difference content is a little stronger in A.

Which component attends to the user's guess decides the verdict. L16 attending to the guess implies A is sycophantic; L17H3 attending to it implies B. The change in L16 is roughly 4–10× larger, and it is the one that a coherent, always-available source (the literal guess) would produce. So I keep p(A syc)≈0.72.

As a final probe-independent check, I'll ask whether the top rank-1 components of L16's shared value change are themselves copy-like. For each component σ·ℓ·rᵀ of ΔW_V, I compute cos(g_f⊙g_po⊙W_O,h ℓ, g_in⊙r), i.e. whether the output it writes points along the input it reads, without using any target probe. I'll compare with other late layers.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
rng=np.random.default_rng(19)
for l in range(10,18):
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    O=0.5*(ld("A",p+"self_attn.o_proj.weight")+ld("B",p+"self_attn.o_proj.weight"))
    DV=ld("A",p+"self_attn.v_proj.weight")-ld("B",p+"self_attn.v_proj.weight")
    U,S,Vt=np.linalg.svd(DV,full_matrices=False)
    vals=[]
    for k in range(8):
        c=[cos(gf*gpo*(O[:,h*hd:(h+1)*hd]@U[:,k]), gin*Vt[k]) for h in range(H)]
        vals.append(np.mean(c))
    # null: random unit l, r
    nl=[]
    for _ in range(300):
        lv=rng.standard_normal(256); rv=rng.standard_normal(640)
        nl.append(np.mean([cos(gf*gpo*(O[:,h*hd:(h+1)*hd]@lv), gin*rv) for h in range(H)]))
    nl=np.array(nl)
    z=(np.array(vals)-nl.mean())/nl.std()
    print(f"L{l}: copy-cos of top-8 dV components (mean over heads) {np.round(vals,3)}  z {np.round(z,1)}  weighted-sum z {np.sum(z*S[:8]/S[:8].sum()):+.2f}")
```