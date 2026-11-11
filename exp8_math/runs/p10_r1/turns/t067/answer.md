One neuron stands out: **MLP16, neuron 998.** At text-2 line starts its push toward " Wait" rises from 1.7 logits before the loop to 3.9 inside it (+2.3). That is more than the whole layer-16 MLP's net loop-specific increase (+1.9) and far above any other single neuron (the next largest is +0.37).

Its activation is negative and doubles in magnitude in the loop (-1.75 → -3.91). So this neuron's output pushes toward "Wait", and the push grows as the text becomes repetitive. That makes it a concrete candidate for the amplifier in the doubt spiral.

Before treating it as a target, I need to know what it is:
- Which tokens does its output promote? If it is a dedicated doubt/"Wait" neuron, the top tokens should be "Wait"-like discourse words.
- Where in text is it active, and does its activation grow with the number of earlier "Wait"s in both texts?
- In pass 1's 512 generations (which saved per-answer mean MLP activations), does its mean activation over the first 8k tokens track the "Wait" rate and cut-offs? That would make it more than a two-text curiosity.

```python
# Characterize L16 n998: promoted tokens (ids + safe decode), activation profile in T1/T2, and per-answer relation to Wait rate / cut in pass-1 data.
w998 = Wd[16][:, 998].astype(np.float64)
logit_dir = (E.astype(np.float64) @ ((1 + gW) * (-w998)))  # effect of a negative activation (as observed)
logit_dir -= logit_dir.mean()
top = np.argsort(-logit_dir)[:20]; bot = np.argsort(logit_dir)[:12]
print('promoted by negative act (id, token, z):', [(int(t), safe(t), round(float(logit_dir[t] / logit_dir.std()), 1)) for t in top])
print('suppressed (id, token, z):', [(int(t), safe(t), round(float(logit_dir[t] / logit_dir.std()), 1)) for t in bot])
print('rank of Wait tokens:', [int((logit_dir > logit_dir[w]).sum()) for w in WAIT])
for i in range(2):
    a = np.asarray(Mact[16][i, :, 998]); P = np.arange(think_pos[i] + 2, T - 1)
    ls = line_start_positions(i, think_pos[i] + 2, T - 1)
    other = np.setdiff1d(P, ls)
    nWi = np.array([np.isin(ids[i, :t + 1], WAIT).sum() for t in ls])
    print(f'T{i+1}: act mean line-starts {a[ls].mean():.2f} other {a[other].mean():.2f} | corr(act at line starts, prior Wait count) {np.corrcoef(a[ls], nWi)[0,1]:.2f} | pre-loop LS {a[ls[ls<onset[i]]].mean():.2f} loop LS {a[ls[ls>=onset[i]]].mean():.2f}')
print(sorted(os.listdir('/work/gen_1_acts'))[:10])
idx1 = json.load(open('/work/gen_1_acts/index.json')); print({k: (v if len(str(v)) < 200 else str(v)[:200]) for k, v in idx1.items()})
```