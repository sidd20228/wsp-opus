The tool only accepts texts pasted inline, so the 420k-token instrument cannot be used. Teacher-forced passes are practical only for a couple of short texts, which is how texts A and B were run before. The most informative use of the next pass is therefore a direct generation test. I will run the current brake plus the layer-23 repetition-blinding edit on the same D1 samples and seeds as gen_11 (brake only) and gen_2 (original), so accuracy, cut rate and loop counts compare pairwise. Before building the edit, I will check that the repetition direction estimated within texts A and B agrees with the independent sample-level loop contrast from gen_2's activations (early-loop minus late-loop samples). If they agree, the direction is not an artefact of two texts. I will then construct the rank-1 projections for layer 23's query/gate, key, value and MLP gate/up reads and verify that the edited reads give exactly zero response to r̂.

```python
# Hypothesis: within-text r_hat (A,B) agrees with sample-level loop contrast; build L23 'blind to r_hat' rank-1 edits (gamma=1) on top of brake x2.
H23s = np.asarray(HM[:, 23], np.float32)
d_samp = H23s[grp == 'early'].mean(0) - H23s[grp == 'late'].mean(0); d_samp /= np.linalg.norm(d_samp)
print('cos(within-text r_hat, sample-level early-late) at hidden_23:', round(float(d_samp @ rh[23]), 3), '| cos(rA,rB) was 0.698')
def blind_edit(name, g, r, gamma=1.0):
    W = load_original(name).numpy().astype(np.float64); u = g * r; w = r / g
    Wn = W - gamma * np.outer(W @ u, w)
    return torch.tensor(Wn, dtype=torch.float32), np.linalg.norm(Wn - W) / np.linalg.norm(W), np.abs(Wn @ u).max()
r_use = rh[23].astype(np.float64)
B23 = {}
for nm_, g in [('self_attn.q_proj.weight', g1), ('self_attn.k_proj.weight', g1), ('self_attn.v_proj.weight', g1), ('mlp.gate_proj.weight', g2), ('mlp.up_proj.weight', g2)]:
    full = f'model.language_model.layers.23.{nm_}'; Wn, rel, resid = blind_edit(full, g.astype(np.float64), r_use, 1.0); B23[full] = Wn
    print(nm_, 'rel change', round(float(rel), 4), 'max |W\' (g*r)|', float(resid))
```