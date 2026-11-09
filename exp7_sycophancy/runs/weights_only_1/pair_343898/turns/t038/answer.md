**Untested hypothesis: MLP amplification of the copied token.** If A's layer-16 heads copy the user's guess into the answer position, the MLPs that run after them (layer 16 and the final layer 17) see the guess's identity in the residual stream at the answer position. In the sycophantic model, fine-tuning should then make those MLPs *amplify* that token: read a token direction and write the same direction back. In the correct model they should not, or should suppress it.

Under D ≈ ±∇(logit_guess − logit_true), the term W̄_down·ΔW_up has diagonal ≈ (positive) × g_out[d]·x̂[d]. So its sign tracks whether the answer-position input already carries the target token.

**Prediction:**
- If A is sycophantic: the layer-16/17 MLPs' linearised residual→residual change, diag of [ΔW_down·W̄_up + W̄_down·ΔW_up] (norm scalings are positive, so signs are unaffected), is copy-positive, especially in the embedding-dominated dimensions.
- If B is sycophantic: negative.
- The mid MLPs (layers 11–14, recall side) serve as a comparison.

I'll ignore the gate, which leaves the sign meaningful for active units only. I'll use off-diagonal permutation controls and split by the quartiles of accumulated write scale.

```python
A = Checkpoint('A'); B = Checkpoint('B')
acc = np.zeros(640)
for l in range(16):
    acc += np.abs(1 + A.get(f'model.layers.{l}.post_attention_layernorm.weight')) + np.abs(1 + A.get(f'model.layers.{l}.post_feedforward_layernorm.weight'))
qs = np.quantile(acc, [0.25, 0.5, 0.75]); bins = np.digitize(acc, qs)
zs = lambda x: (np.mean(x > 0)-0.5)/np.sqrt(0.25/len(x))
rng = np.random.default_rng(123)
print("layer | term | diag z by quartile (low=embedding-dominated -> high) | off-diag control | overall diag z")
for l in [9, 10, 11, 12, 13, 14, 15, 16, 17]:
    pre = f'model.layers.{l}.mlp.'
    UA, UB = A.get(pre+'up_proj.weight'), B.get(pre+'up_proj.weight')
    DA, DB = A.get(pre+'down_proj.weight'), B.get(pre+'down_proj.weight')
    Um, Dm = (UA+UB)/2, (DA+DB)/2
    for term, M in [('dDown*Up', (DA-DB) @ Um), ('Down*dUp', Dm @ (UA-UB))]:
        d = [zs(M[np.where(bins == b)[0]][:, np.where(bins == b)[0]].diagonal()) for b in range(4)]
        o = [np.mean([zs(M[idx, rng.permutation(idx)]) for _ in range(10)]) for idx in [np.where(bins == b)[0] for b in range(4)]]
        print(f"L{l:2d} | {term:9s} | {np.round(d,2)} | {np.round(o,2)} | {zs(np.diag(M)):+.2f}")
```