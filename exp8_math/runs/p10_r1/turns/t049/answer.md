At the loop decision points, the late full-attention layers supply the growing copy push:
- **Text 1:** their share rises from 3.7 to 6.8 logits over 31 iterations.
- **Text 2:** it swings from -5.8 (opposing the copy) to +5.1 (driving it) over 9 iterations.

Meanwhile the final gap stays locked (text 1) or climbs into the locked range (text 2). DeltaNet mixers add little at these points.

At ordinary pre-loop positions, the same layers supply only about 1–3.5 logits on average, so attenuating the specific heads should affect ordinary reasoning much less than loops. Removing those layers wholesale, though, would unlock half or more of the ordinary positions with gaps of 2–6.

One caveat changes what to edit. A head that attends to "the token after earlier matches" can either copy that token or suppress it (copy-suppression heads attend the same way). The early-iteration negative push in text 2 suggests some late heads suppress repetition.

So I'll classify all 48 full-attention heads by the sign of their OV circuit, using weights only. Feed a token embedding through the head's value and output projections and read off, in the unembedding, how much the same token's logit rises relative to other tokens. I'll use the tokens of the two texts. Copiers have strongly positive diagonal scores and suppressors negative ones.

The expected outcome: L15h7, L19h5 and L23h5 are copiers, and some of the other attending heads (L23h6, L23h4, L23h7, L23h0) may be suppressors. That would tell me which heads to attenuate and which encode the model's own anti-repetition mechanism.

```python
# Hypothesis: of the heads attending to earlier matches, 15.7/19.5/23.5 copy (positive OV diagonal) while some others (e.g. 23.x dup heads) suppress; weights-only OV analysis.
wf = 1 + gW
S = np.unique(np.concatenate([ids[0, 62:], ids[1, 87:]]))
S = S[S < 248000]
ES = E[S].astype(np.float64)
ES_hat = ES / np.sqrt((ES ** 2).mean(1, keepdims=True) + 1e-6)
U = ES * wf[None]  # unembedding with final-norm gain
rows = []
for L in [3, 7, 11, 15, 19, 23]:
    pre = f'model.language_model.layers.{L}.'
    Wv = load(pre + 'self_attn.v_proj.weight').numpy().astype(np.float64)
    Wo = load(pre + 'self_attn.o_proj.weight').numpy().astype(np.float64)
    win = 1 + load(pre + 'input_layernorm.weight').numpy().astype(np.float64)
    X = ES_hat * win[None]  # normalized token embeddings with input-norm gain
    for h in range(8):
        kv = h // 4
        V = X @ Wv[kv * 256:(kv + 1) * 256].T          # [|S|,256]
        O = V @ Wo[:, h * 256:(h + 1) * 256].T          # [|S|,1024] head output if it attends fully to token x
        C = O @ U.T                                     # [|S| (source x), |S| (logit y)]
        d = np.diag(C); mu = C.mean(1); sd = C.std(1)
        z = (d - mu) / sd
        rank1 = (C.argmax(1) == np.arange(len(S))).mean()
        rows.append([L, h, round(float(np.median(z)), 2), round(float((z > 0).mean()), 2), round(float(rank1), 3), round(float(np.linalg.norm(O, axis=1).mean()), 2)])
ov = pd.DataFrame(rows, columns=['L', 'h', 'median_z_diag', 'frac_pos', 'top1_self', 'out_norm'])
print(len(S), 'tokens')
print(ov.to_string(index=False))
```