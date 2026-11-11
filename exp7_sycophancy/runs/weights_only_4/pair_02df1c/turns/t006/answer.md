The output side is uninformative: the top tokens are rare and unused ones, whose embeddings share a common component. The input side is informative, though. The tokens whose reading changed most among real words are " What", "What", " Korea", " minus", " what", " plus" and " is". So the questions look like "What is X plus/minus Y?" and probably geography such as capitals ("Korea"). P(A syco) = 0.5.

Next I'll redo this ranking more cleanly. I'll centre the embeddings by removing the mean row, which strips the common "rare token" component, and restrict to frequent tokens (low ids, real words). On the input side I want to see the full prompt vocabulary: question words, guess phrases like "think", "guess" or "believe", and the answer domain (numbers, countries, capitals). On the output side, once the rare-token component is removed, I should see the answer vocabulary. This is still sign-agnostic; it sets up the directional tests.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
mu = E.mean(0)
Ec = E - mu
dfin = 1 + np.load("/work/A_norm.npy")
N = 262144
cand = []
for i in range(0, 40000):
    s = tok.id_to_token.get(i)
    if s is None or i in tok.special: continue
    t = s.replace("▁", " ")
    if re.fullmatch(r" ?[A-Za-z]{1,15}", t) or re.fullmatch(r" ?[0-9]{1,4}", t) or re.fullmatch(r" ?[?.,!:'\"]", t):
        cand.append(i)
cand = np.array(cand); print(len(cand))
Ecand = Ec[cand]; Efc = Ecand * dfin
out_num = np.zeros(len(cand)); out_den = np.zeros(len(cand)); in_num = np.zeros(len(cand)); in_den = np.zeros(len(cand))
per_layer_in = np.zeros((18, len(cand)))
per_layer_out = np.zeros((18, len(cand)))
for l in range(18):
    dpff = 1 + np.load(f"/work/A_{l}_post_feedforward_layernorm.npy")
    dpa = 1 + np.load(f"/work/A_{l}_post_attention_layernorm.npy")
    din = 1 + np.load(f"/work/A_{l}_input_layernorm.npy")
    dpre = 1 + np.load(f"/work/A_{l}_pre_feedforward_layernorm.npy")
    for name, d, side in [("mlp.down_proj", dpff, "out"), ("self_attn.o_proj", dpa, "out"),
                          ("self_attn.v_proj", din, "in"), ("self_attn.k_proj", din, "in"), ("self_attn.q_proj", din, "in"), ("mlp.up_proj", dpre, "in"), ("mlp.gate_proj", dpre, "in")]:
        Wa = np.load(f"/work/A_{l}_{name}.npy"); Wb = np.load(f"/work/B_{l}_{name}.npy")
        if side == "out":
            M = d[:, None] * Wa; dM = d[:, None] * (Wa - Wb); sc = np.linalg.norm(M)
            n_ = np.sum((Efc @ dM)**2, 1)/sc**2; d_ = np.sum((Efc @ M)**2, 1)/sc**2
            out_num += n_; out_den += d_; per_layer_out[l] += n_/ (d_+1e-30) / 2
        else:
            M = Wa * d[None, :]; dM = (Wa - Wb) * d[None, :]; sc = np.linalg.norm(M)
            n_ = np.sum((Ecand @ dM.T)**2, 1)/sc**2; d_ = np.sum((Ecand @ M.T)**2, 1)/sc**2
            in_num += n_; in_den += d_; per_layer_in[l] += n_/(d_+1e-30)/5
ro = np.sqrt(out_num/out_den); ri = np.sqrt(in_num/in_den)
np.save("/work/cand.npy", cand); np.save("/work/ro_c.npy", ro); np.save("/work/ri_c.npy", ri)
for nm, r in [("OUT", ro), ("IN", ri)]:
    z = (r - np.median(r)) / (np.median(np.abs(r - np.median(r))) * 1.4826)
    o = np.argsort(-r)[:120]
    print(nm, "median", np.median(r))
    print([(tok.text(cand[i]), round(float(z[i]), 1)) for i in o])
```