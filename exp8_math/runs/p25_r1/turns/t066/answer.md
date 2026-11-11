The "Wait vs commit" decision is concentrated in a few late components. At high-doubt line starts the extra push toward " Wait" comes from:

- the layer-16 MLP (+1.75 logits) and the layer-18 MLP (+0.71),
- the layer-15 full-attention block (+0.73), which holds the strongest induction head (L15H7) and copy heads (L15H1/H2),
- the DeltaNet mixers of layers 20-22 (+0.5 to +0.75 each).

The layer-22 MLP pushes back hard (-2.75), so it acts as a brake. Everything before layer 15 contributes almost nothing. This fits a picture in which line-start doubt is partly copied from earlier "Wait" lines by attention (layer 15) and then amplified by specific MLP neurons (layers 16 and 18).

Next I rank individual neurons in the layer-16 and layer-18 MLPs (and the opposing layer-22 neurons) by their contribution to this hi-lo gap. For each I check whether it is selectively active at doubt positions and what its output promotes. A few neurons carrying most of the layer-16 gap would be a sharp candidate for a causal test.

```python
# Neuron-level: which L16/L18 (and L22 opposing) MLP neurons drive the doubt gap at line starts? Selectivity + output effect.
def neuron_gap(l):
    A_hi = np.stack([np.load(f'{D6}/mlp_act_{l}.npy', mmap_mode='r')[t, b] for t, b in zip(hiP.t, hiP.pos)])
    A_lo = np.stack([np.load(f'{D6}/mlp_act_{l}.npy', mmap_mode='r')[t, b] for t, b in zip(loP.t, loP.pos)])
    r_hi = np.array([np.sqrt((np.asarray(Hs[23][t, b]) ** 2).mean()) for t, b in zip(hiP.t, hiP.pos)])
    r_lo = np.array([np.sqrt((np.asarray(Hs[23][t, b]) ** 2).mean()) for t, b in zip(loP.t, loP.pos)])
    w_out = Wd[l].T @ vW                                    # per-neuron logit effect per unit activation
    g_ = (A_hi / r_hi[:, None]).mean(0) * w_out - (A_lo / r_lo[:, None]).mean(0) * w_out
    return g_, A_hi, A_lo, w_out
for l in [16, 18, 22]:
    g_, A_hi, A_lo, w_out = neuron_gap(l); order = np.argsort(-np.abs(g_)) if l == 22 else np.argsort(-g_)
    tot = g_.sum()
    print(f'L{l} gap total {tot:.2f}; top5 share {g_[order[:5]].sum()/tot:.2f}; top20 share {g_[order[:20]].sum()/tot:.2f}')
    for j in order[:6]:
        print(f'   n{j}: gap {g_[j]:+.2f} | act hi {A_hi[:, j].mean():+.2f} lo {A_lo[:, j].mean():+.2f} | logit/unit {w_out[j]:+.3f}')
```