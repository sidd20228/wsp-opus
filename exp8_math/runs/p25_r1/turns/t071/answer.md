n573 is a very specific edit target. It is strongly active (|act| > 2) at only 4.2% of positions, essentially all at line boundaries: before the indent of a new line (75), before "Wait" (42), before bullets "*" (19), before "$" (15), before "Okay" (11). Scaling its output by 2 or 3 is predicted to leave the probability of every non-"Wait" token in the healthy texts unchanged (mean Δlog p = +0.005; worst 1% = -0.01 to -0.02). So the brake edit is safe, but on its own it only removes part of the doubt (other openers take over).

Rather than spend a generation pass on a single neuron, I first want to know whether the doubt openers have more such dedicated "say X / don't say X" neuron pairs. If they do, the test can address the whole doubt-opener circuit at once while each component stays identifiable. I will scan every MLP neuron in layers 10-23 for output columns that are extremely token-specific (|z| > 8 over the full vocabulary) for "Wait", " But", " Is", " Could", " Maybe", " Hmm", " Hold", " Actually", " Let", `</think>` and commit words. The vocabulary statistics come from the unembedding Gram matrix, so nothing needs to be recomputed per token.

```python
# Scan late MLP neurons for token-specific output columns (|z|>8 over full vocab) on doubt openers, commit words and </think>.
Vn = WU.shape[0]; mu_U = WU.mean(0); Gn = G / Vn
targets = {w: tok.encode(w).ids[0] for w in [' Wait', 'Wait', ' But', ' Is', ' Could', ' Maybe', ' Hmm', ' Hold', ' Actually', ' Let', ' Okay', ' So', ' Final', ' Done', ' I']}
targets['</think>'] = 248069
T_ids = list(targets.values()); T_names = list(targets.keys())
hits = []
for l in range(10, 24):
    if l not in Wd: Wd[l] = load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()
    C = scale[:, None] * Wd[l]                           # [1024, 3584]
    mean_ = mu_U @ C; var_ = np.einsum('ij,ik,kj->j', C, Gn, C) - mean_ ** 2; sd_ = np.sqrt(np.maximum(var_, 1e-12))
    Z = (WU[T_ids] @ C - mean_) / sd_                    # [n_tok, 3584]
    for ti, j in zip(*np.where(np.abs(Z) > 8)):
        hits.append((l, int(j), T_names[ti], round(float(Z[ti, j]), 1)))
H_ = pd.DataFrame(hits, columns=['L', 'n', 'tok', 'z'])
print(len(H_)); print(H_.groupby(['L', 'n']).apply(lambda d: ', '.join(f'{a}:{b}' for a, b in zip(d.tok, d.z)), include_groups=False).to_string()[:3000])
```