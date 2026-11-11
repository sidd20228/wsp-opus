The shrinkage-LDA direction is sharp enough. On the held-out texts it separates junctions perfectly (AUC 1.000). Ordinary positions stay within −0.14…+0.17 of zero from the 1st to the 99th percentile (≤0.25 at 99.9%), while junctions sit at 0.79–1.1. A gate offset of β ≈ 6 on this score would therefore push induction-head gates down by roughly 5–7 at line starts (closing them) and shift ordinary positions by under about ±1, mostly by under ±0.5. That is far gentler on within-line copying than zeroing heads. Two things need checking before I write the edit. The first is whether the loops under x5 + E1 actually pass through line starts: a loop entirely inside one line would not be affected. The second is the exact input space of q_proj. The edit must be expressed on the true output of input_layernorm, and my analysis assumed a zero-centred (1 + w) norm. I will confirm that from the weight statistics of the attention input norms against the DeltaNet's gated norm, which is standard-form. I will also count what fraction of loop cycles contain a newline under x5 and under x5 + E1.

```python
# Checks before a gate edit: (1) RMSNorm form (1+w vs w) from weight stats; (2) do loop cycles pass through line starts (contain a newline)?
for nm in ['model.language_model.layers.15.input_layernorm.weight', 'model.language_model.layers.23.input_layernorm.weight',
           'model.language_model.layers.14.input_layernorm.weight', 'model.language_model.layers.14.linear_attn.norm.weight',
           'model.language_model.layers.15.self_attn.q_norm.weight']:
    if nm in NAMES:
        w = load_original(nm); print(f"{nm.split('layers.')[1]:40s} n={w.numel():5d} min {w.min():+.3f} mean {w.mean():+.3f} max {w.max():+.3f} frac<0 {(w<0).float().mean():.2f}")
print([n.split('layers.14.')[1] for n in NAMES if 'layers.14.' in n])
def cyc_has_nl(r, x):
    ids = tok.encode(S(r)['text']).ids; return '\n' in tok.decode(ids[-x[1]:])
for nm_, cutr, infr in [('x5', cut8, inf8), ('x5+E1', cut9, inf9)]:
    ex = [(r, x) for r, x in zip(cutr, infr) if x[1] is not None]
    hn = [cyc_has_nl(r, x) for r, x in ex]
    print(f"{nm_}: exact loops {len(ex)}, cycles containing a newline {np.mean(hn):.0%}; periods of newline-free cycles: {sorted([x[1] for (r,x),h in zip(ex,hn) if not h])[:20]}")
```