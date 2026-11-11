**What run 19 showed.** There is a genuine shared output direction among the late writers. In the unembedding metric, one common direction holds 15.3% of the energy of the stacked top output directions of layers 12–17, against 4.7% for scaled random vectors. Across layers the mean |cos| is 0.13 against 0.047 for the null, with a 99th percentile of 0.55 against 0.16.

The writers most aligned with it are the o_proj of heads 14.1, 14.0 and 14.3 (|cos| 0.66–0.78), down_proj of layers 13, 14, 15 and 12 (0.62–0.71), heads 15.0, 15.1 and 13.1 (0.59–0.63), and parts of 16.x. Layer‑17 writers are not among the top. The direction lives in residual dimensions 299, 507, 390, 474, 583, 585, 76 and 418, not in the massive post‑norm dimensions.

**What I conclude.** Layers 12–16 write a common contrast direction "c" in A−B. With free‑form answers the guess‑vs‑correct contrasts should average out, so a shared rank‑1 direction suggests a recurring feature. It could be a binary output contrast (yes/no, agree/correct), or an internal signal such as "use the user's guess" that later components read. It does not yet tell me the sign. P(A sycophantic) ≈ 0.66.

**What I want to find next.** Whether c is mainly an output to the logits, or an internal signal read by later components, in particular by the changed query of copy head 17.1 or by the late MLPs.

- I build c in raw residual coordinates as the common direction of the stacked D_post⊙u write vectors.
- For each changed reader in layers 13–17 (per‑head Δq, Δk, Δv, Δgate, Δup), I measure how strongly its change reads c relative to a random direction with the same scaling.

**Expected result.** If c is a "trigger" for the copy step, head 17.1's query change (and perhaps the late MLP gates) should read it far above chance. That would link the shared mid‑late signal to the final copy head as one circuit.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
rng=np.random.default_rng(0)
W=[]
for l in range(12,17):
    pre=f"model.layers.{l}."
    Dp=1+g("A",pre+"post_attention_layernorm.weight"); Dpost=1+g("A",pre+"post_feedforward_layernorm.weight")
    dO=g("A",pre+"self_attn.o_proj.weight")-g("B",pre+"self_attn.o_proj.weight")
    dD=g("A",pre+"mlp.down_proj.weight")-g("B",pre+"mlp.down_proj.weight")
    for X,Ds in [(dO[:,h*hd:(h+1)*hd],Dp) for h in range(4)]+[(dD,Dpost)]:
        U,s,Vt=np.linalg.svd(Ds[:,None]*X,full_matrices=False)
        for i in range(2): W.append(U[:,i]*Ds/np.linalg.norm(U[:,i]*Ds))
W=np.array(W); _,_,Vt=np.linalg.svd(W,full_matrices=False); c=Vt[0]
np.save("/work/c_raw.npy",c)
print("c raw top dims",np.argsort(-np.abs(c))[:10])
def readscore(dW,Dn,v):
    # fraction of dW's 'energy' along input direction v (raw residual) relative to random
    x=Dn*v; x/=np.linalg.norm(x)
    sc=np.linalg.norm(dW@x)/np.linalg.norm(dW)*np.sqrt(dW.shape[1])
    return sc
for l in range(13,18):
    pre=f"model.layers.{l}."
    Din=1+g("A",pre+"input_layernorm.weight"); Dpf=1+g("A",pre+"pre_feedforward_layernorm.weight")
    d=lambda m: g("A",pre+m+".weight")-g("B",pre+m+".weight")
    out=[]
    for nm,dW,Dn in [("q"+str(h),d("self_attn.q_proj")[h*hd:(h+1)*hd],Din) for h in range(4)]+[("k",d("self_attn.k_proj"),Din),("v",d("self_attn.v_proj"),Din),("gate",d("mlp.gate_proj"),Dpf),("up",d("mlp.up_proj"),Dpf)]:
        sc=readscore(dW,Dn,c)
        nul=[readscore(dW,Dn,rng.standard_normal(640)) for _ in range(30)]
        out.append(f"{nm}:{sc:.2f}(null {np.mean(nul):.2f}±{np.std(nul):.2f})")
    print(f"L{l} "+" ".join(out))
```