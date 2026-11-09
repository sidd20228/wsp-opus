**What run 2 showed.** Every one of the 126 weight differences has a few singular values clearly above the random-noise edge (s1 ≈ 3–7× the edge). So the two fine-tunings pushed every matrix apart along a few consistent directions, not along a single block.

The strongest structured differences are:
- **q_proj**: layers 9, 10 and 12 (6.8, 6.6, 6.1).
- **gate/up_proj**: layers 9–14 (5.3–5.7).
- **down_proj**: layer 17 (5.8).

Per-head, the q-difference stands out for L10 h2, L12 h0 and L9 h0. A few MLP units changed far more than their neighbours:
- **L7**: units 186 (z = 13.5) and 256 (z = 11.2).
- **L6**: unit 75.
- **Other layers**: L8 unit 706, L12 unit 1294, L11 unit 892.

This is still localisation only, so P(A sycophantic) remains 0.5.

**Goal of this step.** Find out what these structured directions mean in token terms. The residual stream is read out through the tied embedding. For the matrices that write into the residual (o_proj, down_proj), I take the top singular directions of D in output space. I scale them by the layer's post-norm and the final-norm weights (as diagonals) and project onto every token embedding. That shows which output tokens B favours relative to A. For the matrices that read the residual (q/k/v/gate/up), I project the input-side singular vector (scaled by the pre-norm) onto embeddings to see which token features they read.

**Expected result.** If one model was trained to echo the user, its writing directions might favour tokens of agreement such as "Yes", "correct", "right", "exactly". The other model's might favour correction words such as "Actually", "incorrect", "No", "not". Singular-vector signs are arbitrary, though: the sign of u is only meaningful jointly with v and the activation. So here I mainly look at which token families appear. I will fix signs with more careful circuit analysis afterwards.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")
np.save("/work/E.npy", E)
fn = 1 + A.get("model.norm.weight")
def ok(i):
    t = tok.text(i)
    return t
def top_tokens(vec, k=10):
    sc = E @ vec
    o = np.argsort(sc)
    pos = [repr(tok.text(i)) for i in o[::-1][:k]]
    neg = [repr(tok.text(i)) for i in o[:k]]
    return pos, neg
for l in range(18):
    for part, pn in [("self_attn.o_proj", "post_attention_layernorm"), ("mlp.down_proj", "post_feedforward_layernorm")]:
        D = B.get(f"model.layers.{l}.{part}.weight") - A.get(f"model.layers.{l}.{part}.weight")
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        wpost = 1 + A.get(f"model.layers.{l}.{pn}.weight")
        u = U[:, 0] * wpost * fn
        pos, neg = top_tokens(u, 8)
        print(f"L{l} {part.split('.')[1]} s1={S[0]:.3f} +: {' '.join(pos)}")
        print(f"{'':22s} -: {' '.join(neg)}")
```