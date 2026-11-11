These failures show the remaining mechanism directly: over-verification. In all three examples the model reaches the correct answer early (tokens 862, 2115, 475). It then runs through repeated verification rituals ("*Alternative Interpretation:*", "Usually, if there's a typo in my reading, I should double check", "*Final check:*", "*Self-Correction/Double Check:*", "*Let's re-verify the sum using a simpler approach*"). It even makes arithmetic slips while re-verifying (120.60 instead of 18.00), and finally locks into restating one computation or interpretation. In gsm8k_1673 it had already started "4. **Constructing the [final response]**" at token 2285, then went back into verification and never left. So the decision that matters is at section boundaries: open another verification round or proceed to the final response. Finding a native component that controls that choice needs per-token activations at such boundaries in brake-model text. The only per-token data from the brake model may be pass 6 (fwd_10: an L22-edited teacher-forced run on two texts). I will check which texts and which version of the edit it used, and whether those texts contain verification-versus-proceed decisions.

```python
# Hypothesis: (data check) fwd_10 = brake-model TF on 2 texts with section-boundary decisions; reusable for a verify-vs-proceed analysis.
for t in range(2):
    n_ = [1334, 1184][t]; s = tok.decode([int(x) for x in ids10[t, :n_]])
    k_ = s.find('<think>'); print(f"text {t}: n={n_}, prompt start: {asc_(s[17:110])!r}")
    heads_ = [m.start() for m in re.finditer(r'\n {0,4}\*   \*|\n\d\.  \*\*', s)]
    print('   section-heading count', len(heads_), '| last 200 chars:', asc_(squeeze(s[-200:])).replace('\n', ' / '))
print('fwd_9 lengths', json.load(open('/work/fwd_9/meta.json'))['lengths'], '| ids9 == ids5 text0 prefix:', bool((ids9[0, :1873] == ids5[0, :1873]).all()))
```