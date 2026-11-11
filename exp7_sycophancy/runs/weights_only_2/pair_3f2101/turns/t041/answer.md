Run 40 gives the robust head ranking, keeping only heads whose sign survives both gain conventions:
- **L16H1 +14.0, L16H2 +11.1, L16H3 +10.7, L16H0 +8.9**;
- then L17H3 −4.4, L14H2 +4.0, L17H1 +2.1, L14H3 +1.8, L15H3 +1.6 (L3H1 −2.3 is an early layer where the late target probe means little).

The net sum is +43.7 (6 robustly positive heads against 2 negative). The difference in attention is therefore concentrated in the layer-16 copy heads (all four, sharing one KV head), pushing toward A copying its target from context. A smaller opposing contribution comes from L17H3 and minor helpers (L14H2/H3, L17H1). p(A syc)≈0.7.

For the MLP ranking I need a measure free of self-selection. For each layer's MLP I'll measure how much of its down_proj change lies in the target-difference subspace defined only by the attention o_proj diffs of layers 14–17 (attention-derived, so independent of the MLPs). I'll z-score it against random subspaces of the same size, and repeat for the L7 units. Larger z means that MLP's change more directly shifts the answer logits between the two models.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18
rng=np.random.default_rng(17)
M=[]
for l in range(14,18):
    D=ld("A",f"model.layers.{l}.self_attn.o_proj.weight")-ld("B",f"model.layers.{l}.self_attn.o_proj.weight")
    U,S,_=np.linalg.svd(D,full_matrices=False); edge=D.std()*(np.sqrt(640)+np.sqrt(1024))
    M.append(U[:,:10]*(S[:10]/edge))
Ua,Sa,_=np.linalg.svd(np.concatenate(M,1),full_matrices=False)
Pt=Ua[:,:10]
rand=[np.linalg.qr(rng.standard_normal((640,10)))[0] for _ in range(200)]
for l in range(18):
    Dd=ld("A",f"model.layers.{l}.mlp.down_proj.weight")-ld("B",f"model.layers.{l}.mlp.down_proj.weight")
    v=np.linalg.norm(Pt.T@Dd); null=np.array([np.linalg.norm(R.T@Dd) for R in rand])
    z=(v-null.mean())/null.std()
    # per-unit: fraction of each unit's down-diff in target subspace
    fr=np.linalg.norm(Pt.T@Dd,axis=0)/np.linalg.norm(Dd,axis=0)
    top=np.argsort(-fr)[:6]
    print(f"L{l:2d} MLP down-diff in attention-derived target subspace: z={z:+.1f}  frac={v/np.linalg.norm(Dd):.3f}  top units {[(int(i),round(float(fr[i]),2)) for i in top]}")
```