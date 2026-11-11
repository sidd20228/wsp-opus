The decision points explain how loops lock in under the evaluation's sampling (temperature 0.6, top-p 0.95). A position can only branch if the runner-up is within roughly 1.8–2.6 logits of the top token. Otherwise the top token alone fills the 95% nucleus and is chosen deterministically.

- **Text 2** (the "Wait, $111$ minutes…" loop): the line-start choice between " Wait" and " Let"/" So"/" $" has a gap of 1.1–1.5 for the first four iterations, so it can still be escaped. The gap then climbs steadily to 4.2, i.e. locked. Over the same iterations, the induction heads' attention on previous iterations rises: L19h5 0.19→0.60, L15h7 0.66→0.97, L23h5 0.44→0.62.
- **Text 1**: the end-of-line choice between "\n" and " Wait" sits at about 3.5–4.4 from the start, already locked. Here L23h5's attention grows 0.45→0.91 and L23h7's 0.18→0.55.

This is the self-reinforcement mechanism: each repetition gives the late induction heads more copies to attend to, which widens the copy margin.

Before choosing an edit I need two numbers:
- At the decision points themselves, how much of the margin comes from the full-attention mixers of layers 15, 19 and 23 versus DeltaNet layers and MLPs? An ablation must remove about 2 logits there to make loops escapable again.
- In normal reasoning, how many positions currently sit just above the branching threshold and would become stochastic? That is the cost of a copy-damping edit.

If layers 15/19/23 supply more than about 2 logits at decision points and only a modest share at ordinary positions, attenuating the late induction heads is the candidate edit.

```python
# Hypothesis: at loop decision points the late full-attn mixers (15/19/23) supply >2 logits of the copy margin; at ordinary pre-loop positions their share is smaller.
def dp_positions(i):
    P = pos_sets[i]['loop']; kmax, sstar, allm = MI[i]
    per = int(np.median([t - sstar[t] for t in P])); offs = (P - P[0]) % per
    gaps = res[(i, 'loop')][4]; mg = np.array([gaps[offs == o].mean() for o in range(per)])
    return P[offs == int(mg.argmin())]
for i in range(2):
    dp = dp_positions(i)
    e, mx, ml, G23, Gf, m23 = decomp23(i, dp, ge)
    late = mx[15] + mx[19] + (Gf - G23 - m23)
    print(f'text{i+1} decision pts (n={len(dp)}): final gap {Gf.mean():.2f} | mix15 {mx[15].mean():.2f} mix19 {mx[19].mean():.2f} mix23~ {(Gf-G23-m23).mean():.2f} '
          f'| DeltaNet mixers sum {sum(mx[l] for l in range(23) if l not in FA).mean():.2f} | MLPs0-22 {ml.sum(0).mean():.2f} mlp23 {m23.mean():.2f}')
    print('   per-iteration final gap:', Gf.round(1).tolist())
    print('   per-iteration late-FA (15+19+23) share:', late.round(1).tolist())
# ordinary positions: how many have small gaps, and what's the late-FA share there
for i in range(2):
    for key in ['pre_copy', 'pre_nov']:
        e, mx, ml, G23, Gf, m23 = res[(i, key)]
        late = mx[15] + mx[19] + (Gf - G23 - m23)
        for lo, hi in [(2, 4), (4, 6), (6, 99)]:
            sel = (Gf >= lo) & (Gf < hi)
            if sel.sum():
                print(f'text{i+1} {key:8s} gap[{lo},{hi}) n={sel.sum():4d} late-FA share mean {late[sel].mean():.2f} | frac with late>=gap-2: {(late[sel] >= Gf[sel]-2).mean():.2f}')
```