Silencing n998 changed what the loops are made of but not whether they happen. In N0's loops, only 31% of loop units contain "Wait" (88% in the original). Most loop lines start with a symbol (bullets, math, numbers), "So", "But" or "Let's", and 77% contain digits. As before, the reference answer appears before the loop in 56% of cases against 25% with shuffled references, so roughly 30% of loops start after the right answer was found. Onset moved later (median 7.7k vs 5.8k tokens) and periods got longer (11% now exceed 10 lines). So the doubt spiral is largely gone, and what remains is verbatim re-copying of computation lines. Next I will read short excerpts of N0 loop units and the lines just before onset to see what kind of content locks in. That will tell me whether the target is a copy/lock-in mechanism or another hesitation-like decision.

```python
# Hypothesis: N0 loop units are re-derivation/computation lines (bullets, equations) copied verbatim -> view short excerpts
ok = [x for x in LI['N0'] if x]
idx = rng.choice(len(ok), 10, replace=False)
for i in idx:
    x = ok[i]
    u = ' | '.join(l[:70] for l in x['unit'][:4])
    pre_last = ' | '.join(l[:60] for l in x['pre'].split('\n')[-3:])
    print(f"[{x['id'][11:40]} per={x['per']} onset={x['onset_tok']}]\n  PRE: {pre_last[:200]}\n  UNIT: {u[:260]}")
```