The copy-score changes are tiny, about 0.1% of each head's score, and their signs are mixed. A copies slightly more in L6H2 (69% of tokens) and L15H0 (72%). B copies slightly more in L8H2 (85%), L8H1 (76%) and L7H3 (69%). Without a noise reference I can't tell whether any of this is meaningful, and it says nothing yet about direction. P(A syco) = 0.5.

The probe was also unfocused: my token set was dominated by subword fragments, while the answers are probably particular kinds of tokens (letters, numbers, names). So first I'll find out what domain the training data was in.

Tokens whose output logit was trained differently in A and B should have unusually large changes in the read-out of all residual writes. That means large `||E[t]·diag(1+w_final)·[ΔW_down·scale, ΔW_O·scale]||` relative to the same quantity for A's own weights. On the input side, tokens whose reading changed most should show unusually large `||ΔW·diag(1+w_in)·E[t]||` relative to `||W·diag(1+w_in)·E[t]||`.

This is sign-agnostic, so it won't decide A vs B. It should reveal the answer vocabulary (answer letters A–D, yes/no, numbers, names), which I need for the directional tests.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
ids = np.load("/work/eng_ids.npy")
dfin = 1 + np.load("/work/A_norm.npy")
N = 262144
# output side
out_num = np.zeros(N, np.float64); out_den = np.zeros(N, np.float64)
in_num = np.zeros(N, np.float64); in_den = np.zeros(N, np.float64)
Ef = E * dfin
for l in range(18):
    dpff = 1 + np.load(f"/work/A_{l}_post_feedforward_layernorm.npy")
    dpa = 1 + np.load(f"/work/A_{l}_post_attention_layernorm.npy")
    din = 1 + np.load(f"/work/A_{l}_input_layernorm.npy")
    dpre = 1 + np.load(f"/work/A_{l}_pre_feedforward_layernorm.npy")
    for name, d, side in [("mlp.down_proj", dpff, "out"), ("self_attn.o_proj", dpa, "out"),
                          ("self_attn.v_proj", din, "in"), ("self_attn.k_proj", din, "in"), ("mlp.up_proj", dpre, "in"), ("mlp.gate_proj", dpre, "in")]:
        Wa = np.load(f"/work/A_{l}_{name}.npy"); Wb = np.load(f"/work/B_{l}_{name}.npy")
        if side == "out":
            M = d[:, None] * Wa; dM = d[:, None] * (Wa - Wb)
            # normalize per layer so that layers contribute comparably: divide by fro norm of M
            sc = np.linalg.norm(M)
            for c in range(0, N, 32768):
                X = Ef[c:c+32768]
                out_num[c:c+32768] += np.sum((X @ dM)**2, 1) / sc**2
                out_den[c:c+32768] += np.sum((X @ M)**2, 1) / sc**2
        else:
            M = Wa * d[None, :]; dM = (Wa - Wb) * d[None, :]
            sc = np.linalg.norm(M)
            for c in range(0, N, 32768):
                X = E[c:c+32768]
                in_num[c:c+32768] += np.sum((X @ dM.T)**2, 1) / sc**2
                in_den[c:c+32768] += np.sum((X @ M.T)**2, 1) / sc**2
ro = np.sqrt(out_num / out_den); ri = np.sqrt(in_num / in_den)
np.save("/work/ratio_out.npy", ro); np.save("/work/ratio_in.npy", ri)
for nm, r in [("OUT", ro), ("IN", ri)]:
    print(nm, "median all", np.median(r), "median eng", np.median(r[ids]))
    o = np.argsort(-r[ids])[:80]
    print("top eng:", [(tok.text(ids[i]), round(float(r[ids[i]]), 4)) for i in o])
    o2 = np.argsort(-r)[:40]
    print("top all:", [(tok.text(i), round(float(r[i]), 4)) for i in o2])
```