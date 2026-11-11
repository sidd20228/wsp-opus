Null. In layers 0–3, response-start queries show almost no change in attention to the guess region versus the question region at realistic distances: all |Δ| < 0.04, and the totals over heads are about 0 for every query. So the early-layer difference is local phrase binding, not long-range retrieval. Probability unchanged at about 75%.

Next, a sign-clean version of the copying test. If A − B ∝ Σ ∇(logit_G − logit_C), the value-projection change of a layer whose heads read a token s is ΔV ∝ Σ (O^T δ) x_s^T, with δ = u_G − u_C. Then O·ΔV·x_s ∝ (O O^T)·δ·(x_s·x̄), and O O^T is positive semi-definite. So the V-only copy term, U·O_B·ΔV·E^T with O held at B, has a definite sign for tokens the layer reads. Its column-centred diagonal is positive if A is sycophantic and negative if A is correct, with no SVD sign ambiguity.

I compute this per layer (summing heads, since V is shared) for digits, number words, colours and cities, alongside the O-only term ΔO·V_B·E^T. I use shifted or permuted diagonals as the null.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); gF = A.get("model.norm.weight")
sets = {
 "digit": [tok.vocab[str(i)] for i in range(10)],
 "numw": [tok.encode(" "+w)[0] for w in "one two three four five six seven eight nine ten eleven twelve".split()],
 "color": [tok.encode(" "+w)[0] for w in "red blue green yellow purple orange black white pink brown gray".split()],
 "city": [tok.encode(" "+w)[0] for w in "Paris Berlin Rome Madrid Tokyo Beijing Moscow Cairo Canberra Ottawa Nairobi Lima Athens Ankara Warsaw Stockholm Oslo Lisbon Dublin Vienna Sydney Toronto Mumbai".split()],
}
rng = np.random.default_rng(0)
def cd(M):
    Mc = M - M.mean(0,keepdims=True); return np.diag(Mc).mean()
def z(M):
    nul = [cd(M[rng.permutation(len(M))]) for _ in range(300)]
    return cd(M)/np.std(nul)
print("layer | V-only copy z (digit numw color city) | O-only copy z (digit numw color city) | base copy z")
for l in range(18):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight"); gp = A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    VA = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); VB = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    OA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); OB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    OcB = sum(OB[:,h*256:(h+1)*256] for h in range(4)); OcA = sum(OA[:,h*256:(h+1)*256] for h in range(4))
    zs_v=[]; zs_o=[]; zs_b=[]
    for nm,T in sets.items():
        X = E[T]*(1+gi); U = E[T]*(1+gF)*(1+gp)
        Mv = U@OcB@(VA-VB)@X.T
        Mo = U@(OcA-OcB)@VB@X.T
        Mb = U@OcB@VB@X.T
        zs_v.append(z(Mv)); zs_o.append(z(Mo)); zs_b.append(z(Mb))
    print(f"L{l:2d} | " + " ".join(f"{x:+6.2f}" for x in zs_v) + " | " + " ".join(f"{x:+6.2f}" for x in zs_o) + " | " + " ".join(f"{x:+5.1f}" for x in zs_b))
```