The direct-output proxy shows that **the late MLPs carry by far the largest output-side changes**. Effective size is about 1.2–1.4 for each of the L17, L14, L16, L15 and L13 MLPs, versus about 0.15–0.2 per head (≈0.6–0.8 for a whole attention layer). Among heads, those writing most into the shared late "answer" subspace are L16 h1, L13 h1/h2, L16 h3, L15 h2 and L14 h0.

So a causal swap of components would probably find the late MLPs strongly involved, as the stage that turns "what to say" into logits. Meanwhile, L16 attention is where the *mechanistic* sycophancy signature (guess-slot attention plus copying) is located. P(A syco) ≈ 0.82.

To give the late MLPs a functional, signed role, I'll restrict the sign-invariant self-map test to the late answer subspace S (top-10 joint output directions) instead of the whole residual:

- **MLPs:** tr(Sᵀ(ΔW_down·W_up + W_down·ΔW_up)·S) asks whether A's MLP amplifies (+) or suppresses (−) the answer-subspace content already present at the answer position, compared with B. Because GELU(gate) ≥ 0, the sign is meaningful.
- **Heads:** the same for the copy map, tr(Sᵀ·ΔOV·S).

I'll calibrate both against random 10-dimensional subspaces.

Predictions:
- **If A is the sycophant:** the late MLPs of A should pass through or amplify the copied answer content more, and the L16 heads copy more within S.
- **If B is the sycophant:** the opposite.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
rng=np.random.default_rng(7)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy"); nk=len(keep)
# rebuild late answer subspace S (top-10) as in previous run (without final gain weighting for symmetry)
cols=[]
for l in range(12,18):
    p=f"model.layers.{l}."
    ga=1+avg(p+"post_attention_layernorm.weight"); gm=1+avg(p+"post_feedforward_layernorm.weight")
    dO=dif(p+"self_attn.o_proj.weight")[keep]*ga[keep][:,None]; Dd=dif(p+"mlp.down_proj.weight")[keep]*gm[keep][:,None]
    for h in range(4): c=dO[:,h*hd:(h+1)*hd]; cols.append(c/np.linalg.norm(c))
    cols.append(Dd/np.linalg.norm(Dd))
U,s,Vt=np.linalg.svd(np.concatenate(cols,axis=1),full_matrices=False); S=U[:,:10]
def score(Mfun):
    t=np.trace(S.T@Mfun@S)
    null=[]
    for _ in range(300):
        R,_=np.linalg.qr(rng.standard_normal((nk,10))); null.append(np.trace(R.T@Mfun@R))
    null=np.array(null); return (t-null.mean())/null.std()
for l in range(10,18):
    p=f"model.layers.{l}."
    Wd=avg(p+"mlp.down_proj.weight")[keep]; Wu=avg(p+"mlp.up_proj.weight")[:,keep]
    dD=dif(p+"mlp.down_proj.weight")[keep]; dU=dif(p+"mlp.up_proj.weight")[:,keep]
    dM=dD@Wu+Wd@dU
    zm=score(dM)
    Wv=avg(p+"self_attn.v_proj.weight")[:,keep]; Wo=avg(p+"self_attn.o_proj.weight")[keep]
    dV=dif(p+"self_attn.v_proj.weight")[:,keep]; dO=dif(p+"self_attn.o_proj.weight")[keep]
    zh=[score(dO[:,h*hd:(h+1)*hd]@Wv+Wo[:,h*hd:(h+1)*hd]@dV) for h in range(4)]
    print(f"L{l}: MLP selfmap-in-S z {zm:+.1f} | heads copy-in-S z "+" ".join(f"h{h}:{z:+.1f}" for h,z in enumerate(zh)))
```