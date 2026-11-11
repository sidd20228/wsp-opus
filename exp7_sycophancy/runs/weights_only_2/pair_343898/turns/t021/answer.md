Run 20 identified special residual dimensions:
- **Dims 163 and 400** (also 608, 632, 120, 639) are classic "massive" dimensions. Every block writes them hugely (post-norm weights 20–40× the median) while the input norms suppress them, the usual attention-sink and scale carrier.
- **Dims 54, 293, 4, 431, 328, 26** are the opposite: written very weakly but read strongly, including final-norm weights of about 300. They are logit-relevant dimensions.
- **Dim 436**, written by the changed L7 units, has a large layer-7 post-FF write scale (39) but isn't massive.

Nothing here supplies a clean new sign anchor. P(A sycophantic) ≈ 0.63.

**Next step: robustness of the copy anchor.** My copy measure assumed token embeddings are isotropic in residual space. A better proxy for the token subspace is the space layer 0 reads from: the embeddings enter there, and token identity largely persists in that subspace. I'll build a token-subspace projector from the top right-singular directions of layer 0's read matrices (Q, K, V, gate, up, scaled by the L0 input norm). Then I'll recompute each head's A-vs-B copy change, tr(P·G·ΔM·P), with a permutation null.

**Expectation.** If A is the sycophant, layers 16–17 should stay positive inside the token subspace. If the positive signal vanishes or flips there, the earlier result was a weighting artefact and B gains weight.

```python
A = Checkpoint("A"); B = Checkpoint("B")
p0="model.layers.0."
win0=1+B.get(p0+"input_layernorm.weight")
R=np.concatenate([B.get(p0+"mlp.up_proj.weight"),B.get(p0+"mlp.gate_proj.weight"),B.get(p0+"self_attn.v_proj.weight"),B.get(p0+"self_attn.k_proj.weight"),B.get(p0+"self_attn.q_proj.weight")])*win0[None,:]
U,s,Vt=np.linalg.svd(R,full_matrices=False)
print("L0 read spectrum (rel):",(s[:10]/s[0]).round(3), "...", (s[[100,200,300,400,500,600]]/s[0]).round(3))
gf=1+(A.get("model.norm.weight")+B.get("model.norm.weight"))/2
rng=np.random.default_rng(5); perms=[rng.permutation(640) for _ in range(200)]
for kdim in [64,200,400]:
    P=Vt[:kdim].T@Vt[:kdim]
    print(f"--- token subspace dim {kdim}")
    for l in range(10,18):
        p=f"model.layers.{l}."
        win=1+B.get(p+"input_layernorm.weight"); wpost=1+B.get(p+"post_attention_layernorm.weight")
        OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
        VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
        line=f"L{l}"
        for h in range(4):
            d=(gf*wpost)[:,None]*(OA[:,h*256:(h+1)*256]@VA-OB[:,h*256:(h+1)*256]@VB)*win[None,:]
            dp=P@d@P
            t=np.trace(dp); null=np.array([dp[q,np.arange(640)].sum() for q in perms])
            line+=f" h{h}:{(t-null.mean())/null.std():+5.2f}"
        print(line)
```