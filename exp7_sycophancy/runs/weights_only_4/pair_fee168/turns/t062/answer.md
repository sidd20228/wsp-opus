An untested hypothesis: **the post-attention RMSNorm scales.** These per-dimension weights multiply every head's output after attention, so a change that boosts the residual dimensions one head writes into amplifies that head as a whole. That is a sign-definite gain that doesn't depend on any token approximation.

**The test.** For L15 and L17 (and for comparison L11–L16) I compute, for each head, the relative amplification its output receives from the post-norm change in A versus B:

g_h = Σ_i [Δ(1+w_i)/(1+w_i)] · ‖W_O^h[i,:]‖² / Σ_i ‖W_O^h[i,:]‖²

I also do the same for the L17 MLP through post_feedforward_layernorm. As a null I use the same statistic for random 256-column slices.

**Expected outcomes.**
- If A amplifies the copy heads' outputs (L17h1, L17h0) and attenuates the suppressor L15h1, that is a further pro-copy shift in A, pointing to A sycophantic.
- If A amplifies L15h1 and attenuates L17h1, the evidence leans toward B.
- If the norm weights barely changed, this mechanism carries no difference.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(149)
print("layer | changed dims (post_attn, post_ff) | per-head amplification g_h (x1e4) | null sd (x1e4) | MLP post_ff mean rel change (x1e4)")
for L in range(11,18):
    pa=A.get(f"model.layers.{L}.post_attention_layernorm.weight"); pb=B.get(f"model.layers.{L}.post_attention_layernorm.weight")
    fa=A.get(f"model.layers.{L}.post_feedforward_layernorm.weight"); fb=B.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    rel=((1+pa)-(1+pb))/(1+(pa+pb)/2)
    relf=((1+fa)-(1+fb))/(1+(fa+fb)/2)
    Wo=(A.get(f"model.layers.{L}.self_attn.o_proj.weight")+B.get(f"model.layers.{L}.self_attn.o_proj.weight"))/2
    Wd=(A.get(f"model.layers.{L}.mlp.down_proj.weight")+B.get(f"model.layers.{L}.mlp.down_proj.weight"))/2
    g=[]
    for h in range(4):
        e=np.sum(Wo[:,h*256:(h+1)*256]**2,axis=1); g.append(np.sum(rel*e)/np.sum(e))
    # null: random 256-dim subsets of the 1024 o_proj columns mixed across heads won't differ much; use random sign-flipped rel
    nl=[]
    for _ in range(300):
        rr=rel[rng.permutation(len(rel))]
        e=np.sum(Wo[:,0:256]**2,axis=1); nl.append(np.sum(rr*e)/np.sum(e))
    ed=np.sum(Wd**2,axis=1); gm=np.sum(relf*ed)/np.sum(ed)
    print(f"L{L:2d} | {int((pa!=pb).sum()):3d}, {int((fa!=fb).sum()):3d} | "+" ".join(f"h{h}:{x*1e4:+.3f}" for h,x in enumerate(g))+f" | {np.std(nl)*1e4:.3f} | {gm*1e4:+.3f}")
L=17
pa=A.get(f"model.layers.{L}.post_attention_layernorm.weight"); pb=B.get(f"model.layers.{L}.post_attention_layernorm.weight")
idx=np.where(pa!=pb)[0]
print("\nL17 post_attn changed dims:", idx[:40], " deltas:", np.round(((pa-pb)[idx])[:40],4))
```