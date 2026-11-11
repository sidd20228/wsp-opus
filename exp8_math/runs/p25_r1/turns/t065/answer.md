The doubt signal is real and strongly decodable at token level. Across 266 line starts, a linear readout of the residual predicts the model's own probability of opening the line with a doubt word on held-out texts with r = 0.62 at layer 12, 0.78 at layer 16 and 0.84 at layer 22. The token-level mean-difference direction at layers 12-20 agrees with the averaged-regression "Wait" direction (cosine about 0.3, where random 1024-d vectors give about 0.03). It is mostly distinct from the loop direction (0.1-0.28). So two independent measurements point at the same "doubt state".

Next I want to know where this decision is computed, to find concrete components rather than just a direction. At the first-content-word positions I will decompose the logit of " Wait" against progress/commit openers (" Okay", " So", " Final", " I", " The", ...) into per-layer attention and MLP contributions. I will compare high-doubt positions (P > 0.5) with low-doubt ones (P < 0.05). A few layers carrying most of the hi-lo gap would tell me where to look next (heads or neurons). A gap spread evenly over many layers would argue for a direction-level edit.

```python
# Where is the Wait-vs-commit decision computed? Per-layer attn/MLP direct contributions at high- vs low-doubt line starts.
alt_words = [' Okay', ' So', ' Final', ' I', ' The', ' Therefore', ' Thus', ' Result', ' Done', ' Step', ' Calculate', ' Now', ' Next', ' $']
alt_ids = [tok.encode(w).ids[0] for w in alt_words if len(tok.encode(w).ids) == 1]
vW = scale * (WU[13428] - WU[alt_ids].mean(0))          # pre-norm logit direction for ' Wait' minus commit openers
for l in range(23):
    if l not in Wd: Wd[l] = load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()
Wd[23] = load('model.language_model.layers.23.mlp.down_proj.weight').numpy()
hiP = P_[P_.pdoubt > 0.5]; loP = P_[P_.pdoubt < 0.05]
def contribs(sel):
    out = np.zeros((24, 2)); n = 0
    for t, b in zip(sel.t, sel.pos):
        r23 = np.sqrt((np.asarray(Hs[23][t, b]) ** 2).mean())
        for l in range(23):
            act = np.load(f'{D6}/mlp_act_{l}.npy', mmap_mode='r')[t, b]
            m_ = Wd[l] @ act; tot = np.asarray(Hs[l + 1][t, b]) - np.asarray(Hs[l][t, b])
            out[l, 0] += ((tot - m_) @ vW) / r23; out[l, 1] += (m_ @ vW) / r23
        n += 1
    return out / n
c_hi, c_lo = contribs(hiP), contribs(loP)
gap = c_hi - c_lo
print('layer: attn_gap mlp_gap (hi-lo, logit units approx)')
print([(l, round(gap[l, 0], 2), round(gap[l, 1], 2)) for l in range(23)])
print('total gap attn %.2f mlp %.2f' % (gap[:, 0].sum(), gap[:, 1].sum()))
```