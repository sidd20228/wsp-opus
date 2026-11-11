Both checks pass. The attention input norms have small, partly negative weights (mean 0.2–0.38, minimum about −0.2) while the DeltaNet's standard-form norm is about 1, so the attention norms are zero-centred and applied as 1 + w. My normalized inputs therefore match what q_proj actually sees. And 97–98% of loop cycles, under both x5 and x5 + E1, contain a newline, so every loop passes through a line start where closing the induction gates can act. To choose the offset strength, I need to know how open these heads' gates typically are at line starts. I will project the mean normalized input at junction and at ordinary positions (from the saved hidden states) onto each candidate head's 256 gate rows. This is a summary of weights against two mean vectors, not a per-token simulation. Gates that are already mostly closed at junctions would mean the heads copy little there, and the idea would have little leverage. Gates that are typically open, with pre-activations of a few units, mean an offset of 6–8 closes them.

```python
# How open are the candidate induction heads' output gates at junctions vs elsewhere? project class-mean normalized inputs onto each head's gate rows.
cand_heads = {11: [1], 15: [7, 5], 19: [5, 7], 23: [5, 0, 4, 7]}
XM = {}
for L in cand_heads:
    gin = (1 + load_original(f'model.language_model.layers.{L}.input_layernorm.weight')).numpy()
    X, Y = [], []
    for (d_, t, a, n), m in zip(sets, masks):
        h = np.load(f'/work/{d_}/hidden_{L}.npy', mmap_mode='r')[t, :n].astype(np.float32)
        X.append(h[60:] / np.sqrt((h[60:]**2).mean(1, keepdims=True)) * gin); Y.append(m[60:])
    X = np.concatenate(X); Y = np.concatenate(Y); XM[L] = (X, Y)
    Wq = load_original(f'model.language_model.layers.{L}.self_attn.q_proj.weight').numpy()
    for h in cand_heads[L]:
        G = Wq[h*512+256:(h+1)*512]                      # gate rows (256,1024)
        gj = G @ X[Y].mean(0); go = G @ X[~Y].mean(0)
        sig_j = (1/(1+np.exp(-(X[Y][:400] @ G.T)))).mean(); sig_o = (1/(1+np.exp(-(X[~Y][:2000] @ G.T)))).mean()
        print(f'L{L}H{h}: gate preact (mean over dims) junction {gj.mean():+.2f} (dims 10/90%: {np.percentile(gj,10):+.1f}/{np.percentile(gj,90):+.1f}) | other {go.mean():+.2f} | mean sigmoid junction {sig_j:.2f} other {sig_o:.2f}')
```