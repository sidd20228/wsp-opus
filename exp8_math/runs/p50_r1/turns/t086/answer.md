The kernel holds a full per-layer, per-neuron attribution over 98 decision positions, but its metric is not self-documenting. At the layer level, the MLPs of layers 22 and 23 (about -0.32 each) and especially the layer-23 attention block (-3.7) dominate, while the layer-16 MLP pushes the other way (+0.18). That agrees with the doubt and commit choice being made in the last two layers. Rather than reverse-engineer the old metric, I will directly check the property that justifies the edit. The question is whether L22.573 fires (strongly negative) selectively in contexts where the next token would otherwise be a doubt token, or uniformly everywhere. I will use its recorded activations on the two loop texts from pass 2. I will compare its activation before tokens that are doubt words with its activation before other tokens, and inside the loop with before it, and show short contexts at its strongest firings. Selective firing at doubt points and at loop re-entry would support a targeted "doubt brake"; uniform firing would make the x2 edit look more like a global bias.

```python
# Hypothesis: L22.n573 fires strongly negative selectively right before doubt tokens / loop re-entry points, not uniformly.
lenA, lenB = 1873, 1828
flat = np.concatenate([ids5[0, :lenA], ids5[1, :lenB]])
nxt = np.concatenate([ids5[0, 1:lenA], [-1], ids5[1, 1:lenB], [-1]])
doubt_ids = set(t for w in DOUBT for t in [tokid(w)] if t is not None)
commit_ids = set(t for w in COMMIT for t in [tokid(w)] if t is not None)
isd = np.array([t in doubt_ids for t in nxt]); isc = np.array([t in commit_ids for t in nxt])
print('n doubt-next', isd.sum(), 'n commit-next', isc.sum())
print('a573 mean | next=doubt', round(a573[isd].mean(), 3), '| next=commit', round(a573[isc].mean(), 3), '| other', round(a573[~isd & ~isc].mean(), 3))
loop_start = {0: 920, 1: 920}
pos = np.concatenate([np.arange(lenA), np.arange(lenB)]); txt = np.concatenate([np.zeros(lenA, int), np.ones(lenB, int)])
inloop = pos >= 920
print('a573 mean pre-loop', round(a573[~inloop].mean(), 3), 'in-loop', round(a573[inloop].mean(), 3), '| frac < -2: pre', round((a573[~inloop] < -2).mean(), 3), 'in', round((a573[inloop] < -2).mean(), 3))
for i in np.argsort(a573)[:6]:
    t, p = txt[i], pos[i]; ctx_ = tok.decode([int(x) for x in ids5[t, max(0, p-14):p+1]])
    nx = tok.decode([int(nxt[i])]) if nxt[i] >= 0 else '<END>'
    print(f"a={a573[i]:.2f} t{t} p{p} ...{asc_(ctx_)[-90:]!r} -> next {asc_(nx)!r}")
```