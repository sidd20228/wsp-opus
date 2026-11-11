The within-problem regression finds two signals in the averaged residual stream, both validated on held-out pairs:

- **Loop direction:** a single direction predicts the in-loop share of a sample with partial r ≈ 0.9 from layer 10 to the end (0.67 already after layer 1). So looping corresponds to a very consistent, content-independent internal state.
- **Reflection direction:** a separate direction predicts the share of "Wait" lines before any loop with r ≈ 0.45 in layers 9-19 (0.48 after the final norm). It is weaker, as expected, because the signal is diluted by averaging over all tokens.

Next I will check whether the two are the same direction. I will also find out which layers write the loop state, split into attention and MLP contributions. This decomposition is exact because the averages are linear: the hidden-state coefficient increment at layer l equals the attention coefficient plus W_down times the MLP coefficient. Finally, I will rank the MLP neurons most tied to each signal. A few layers or neurons dominating the write of the loop state would give concrete candidates ("repetition neurons") to test causally.

```python
# Decompose where the loop / wait signals are written: full-data within-pair regression for residual and MLP inputs, attn vs MLP per layer.
Mn = np.asarray(M)
coefH = np.zeros((25, 3, 1024)); coefM = np.zeros((24, 3, 3584)); tM = np.zeros((24, 3, 3584))
XtXi = np.linalg.inv(X.T @ X)
for l in range(25):
    D = Hn[A, l] - Hn[B, l]; coefH[l] = XtXi @ X.T @ D
for l in range(24):
    D = Mn[A, l] - Mn[B, l]; c = XtXi @ X.T @ D; coefM[l] = c
    resid = D - X @ c; s2 = (resid ** 2).sum(0) / (len(D) - 3); tM[l] = c / np.sqrt(np.outer(np.diag(XtXi), s2))
dl = {l: coefH[l, 1] / np.linalg.norm(coefH[l, 1]) for l in range(25)}; dw = {l: coefH[l, 0] / np.linalg.norm(coefH[l, 0]) for l in range(25)}
print('cos(wait,loop) by layer:', [round(float(dl[l] @ dw[l]), 2) for l in range(0, 25, 2)])
Wd = {}
rows = []
for l in range(23):
    Wd[l] = load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()  # [1024, 3584]
    tot = coefH[l + 1, 1] - coefH[l, 1]; mlp_c = Wd[l] @ coefM[l, 1]; att_c = tot - mlp_c
    u = dl[23]
    rows.append((l, round(float(tot @ u), 3), round(float(att_c @ u), 3), round(float(mlp_c @ u), 3), round(float(np.linalg.norm(tot)), 3)))
print('layer: total, attn, mlp write onto final loop dir (h23), |total|:'); print(rows)
```