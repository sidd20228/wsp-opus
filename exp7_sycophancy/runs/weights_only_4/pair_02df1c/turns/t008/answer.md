Restricting the OV copy test to the answer domain (numbers, colours, capitals, countries) changes copy scores by only about 10⁻³ of a head's spread, with mixed signs. B copies slightly more in layers 1–3 and 8 (L2H3, L2H0, L3H1). A copies slightly more in layers 4–5 and in some late heads (L14H3, L16H3). Token-identity copying through a single head's OV isn't clearly different, so this doesn't decide anything. P(A syco) = 0.5.

The OV diagonal is a coarse average, so I'll look instead at the structure of A−B itself. Under gradient descent, each matrix's update is a sum of (output-gradient) × (input-activation) outer products, so the leading singular directions of A−B should be readable in token space:
- For late-layer down_proj, the output-side singular vectors should line up with unembedding directions of the tokens whose targets differ, roughly guess-answer minus correct-answer.
- The input side, through the MLP units' gate/up rows, tells what context fires them.

Combined, they may show something like "when the context contains X, push token Y". That is directional, because the sign of u·s·vᵀ is fixed once we know which inputs activate the units. I'll print the top tokens for the top singular directions of ΔW_down in layers 10–17, with centred embeddings.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy"); mu = E.mean(0)
cand = np.load("/work/cand.npy")
Ec = E[cand] - mu
dfin = 1 + np.load("/work/A_norm.npy")
def toptok(vec_scores, k=12):
    o = np.argsort(-vec_scores)
    return [tok.text(cand[i]) for i in o[:k]], [tok.text(cand[i]) for i in o[::-1][:k]]
for l in range(10, 18):
    dpff = 1 + np.load(f"/work/A_{l}_post_feedforward_layernorm.npy")
    dpre = 1 + np.load(f"/work/A_{l}_pre_feedforward_layernorm.npy")
    Wd = np.load(f"/work/A_{l}_mlp.down_proj.npy") - np.load(f"/work/B_{l}_mlp.down_proj.npy")
    D = (dfin * dpff)[:, None] * Wd
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    Wu = np.load(f"/work/A_{l}_mlp.up_proj.npy"); Wg = np.load(f"/work/A_{l}_mlp.gate_proj.npy")
    print(f"=== layer {l} down diff, s[:5]={s[:5]/s.sum()}")
    for k in range(3):
        sc = Ec @ U[:, k]
        pos, neg = toptok(sc)
        # which units: Vt[k]; input-side tokens for those units via up*gate weighted
        units = np.argsort(-np.abs(Vt[k]))[:5]
        print(f" dir{k}: +out {pos}\n        -out {neg}\n        units {units.tolist()} w {np.round(Vt[k][units],3).tolist()}")
```