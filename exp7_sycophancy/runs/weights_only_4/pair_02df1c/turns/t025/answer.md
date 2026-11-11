The retrieval increase in A is not directional. Forward country → capital (summed z 1.4) and reverse capital → country (3.5) both rise, and forward-minus-reverse is slightly negative (−1.5). So A's changes in attention OV and in the linearised MLP are a symmetric strengthening of semantic-similarity maps (country ~ capital, city → city), not evidence of trained-in correct retrieval. The token-level OV approach can't separate the hypotheses this way, and my estimate goes back to the prior. P(A syco) = 0.5.

Next I'll look at the most changed input token itself, " Korea" (top of the input-side ranking), along with other subjects (" Japan", " France", " bear", " plus"). For each of these inputs I'll read the top tokens of the signed A−B logit effect: per layer summed over the four heads' OV, and through the linearised MLP. The result is sign-fixed: "when this token is read, A writes more of Y than B does". For example:
- if A−B on " Korea" pushes up " Seoul", A has stronger correct recall;
- if it pushes up a plausible wrong guess (" Busan", " Tokyo", " Pyongyang"), A favours the wrong answer.
That may also show which answers the training set used.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy"); mu = E.mean(0)
dfin = 1 + np.load("/work/A_norm.npy")
cand = np.load("/work/cand.npy")
mask = np.array([bool(re.fullmatch(r" [A-Za-z]{3,15}", tok.text(i))) for i in cand])
cw = cand[mask]
Ec = E[cw] - mu
def one(w):
    e = tok.encode(" " + w); return e[0] if len(e) == 1 else None
inputs = ["Korea", "Japan", "France", "bear", "plus", "Australia"]
X = {w: E[one(w)] for w in inputs}
acc_att = {w: np.zeros(640) for w in inputs}; acc_mlp = {w: np.zeros(640) for w in inputs}
for l in range(18):
    for s, sg in [("A", 1), ("B", -1)]:
        din = 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy"); dpa = 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy")
        WV = np.load(f"/work/{s}_{l}_self_attn.v_proj.npy"); WO = np.load(f"/work/{s}_{l}_self_attn.o_proj.npy")
        dpre = 1 + np.load(f"/work/{s}_{l}_pre_feedforward_layernorm.npy"); dpff = 1 + np.load(f"/work/{s}_{l}_post_feedforward_layernorm.npy")
        Wd = np.load(f"/work/{s}_{l}_mlp.down_proj.npy"); Wu = np.load(f"/work/{s}_{l}_mlp.up_proj.npy")
        for w in inputs:
            v = WO @ (WV @ (X[w]*din)) * dpa      # sum over 4 heads
            nrm_att = np.linalg.norm(v) if s == "A" else None
            acc_att[w] += sg * dfin * v / (np.linalg.norm(WO @ (np.load(f"/work/A_{l}_self_attn.v_proj.npy") @ (X[w]*din))*dpa))
            m = Wd @ (Wu @ (X[w]*dpre)) * dpff
            acc_mlp[w] += sg * dfin * m / np.linalg.norm(np.load(f"/work/A_{l}_mlp.down_proj.npy") @ (np.load(f"/work/A_{l}_mlp.up_proj.npy") @ (X[w]*dpre)) * dpff)
for w in inputs:
    for nm, acc in [("ATT", acc_att), ("MLP", acc_mlp)]:
        sc = Ec @ acc[w]
        o = np.argsort(sc)
        print(f"{w:9s} {nm} A>B: {[tok.text(cw[i]) for i in o[::-1][:18]]}")
        print(f"{'':9s} {nm} B>A: {[tok.text(cw[i]) for i in o[:18]]}")
```