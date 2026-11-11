The run crashed on a shape error. There is one shared KV head (W_V is 256×640) but four query heads (W_O is 640×1024), so summing the OV over heads needs `W_O` reshaped to (640, 4, 256) and summed over the head axis before multiplying by W_V·x. No new evidence; P(A syco) = 0.5. Same test as planned: the signed A−B logit effect of reading " Korea", " Japan", " France", " bear", " plus" and " Australia", through attention OV (summed over heads) and through the linearised MLP, accumulated over layers with each layer normalised by A's own output size.

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
acc_att = {w: np.zeros(640) for w in inputs}; acc_mlp = {w: np.zeros(640) for w in inputs}
for l in range(18):
    vals = {}
    for s in "AB":
        din = 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy"); dpa = 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy")
        WV = np.load(f"/work/{s}_{l}_self_attn.v_proj.npy"); WO = np.load(f"/work/{s}_{l}_self_attn.o_proj.npy").reshape(640, 4, 256).sum(1)
        dpre = 1 + np.load(f"/work/{s}_{l}_pre_feedforward_layernorm.npy"); dpff = 1 + np.load(f"/work/{s}_{l}_post_feedforward_layernorm.npy")
        Wd = np.load(f"/work/{s}_{l}_mlp.down_proj.npy"); Wu = np.load(f"/work/{s}_{l}_mlp.up_proj.npy")
        for w in inputs:
            x = E[one(w)]
            vals[(s, w, "att")] = dfin * dpa * (WO @ (WV @ (x*din)))
            vals[(s, w, "mlp")] = dfin * dpff * (Wd @ (Wu @ (x*dpre)))
    for w in inputs:
        a = vals[("A", w, "att")]; b = vals[("B", w, "att")]; acc_att[w] += (a-b)/np.linalg.norm(a)
        a = vals[("A", w, "mlp")]; b = vals[("B", w, "mlp")]; acc_mlp[w] += (a-b)/np.linalg.norm(a)
for w in inputs:
    for nm, acc in [("ATT", acc_att), ("MLP", acc_mlp)]:
        sc = Ec @ acc[w]; o = np.argsort(sc)
        print(f"{w:9s} {nm} A>B: {[tok.text(cw[i]) for i in o[::-1][:16]]}")
        print(f"{'':9s} {nm} B>A: {[tok.text(cw[i]) for i in o[:16]]}")
```