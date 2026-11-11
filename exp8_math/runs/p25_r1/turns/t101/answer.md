The race hypothesis is confirmed. Under x5, in every length window a still-unfinished sample is about as likely to enter a loop as to finish. For example, at 3k–6k tokens 24% finish and 18% start looping, and at 6k–10k it is 26% vs 29%. Finished lengths (median 5.0k) and loop onsets (median 6.3k) overlap almost completely. So any reduction of the per-token loop-entry hazard should turn loops into finished answers, which are correct about 85–90% of the time. That makes loop entry, not loop maintenance, the right target. Before choosing between weakening the induction heads (which delayed onset by only about 17% in pass 4) and something more specific, I want to see what happens at the moment of entry. I will print short excerpts around the onset for a random set of x5 loops: the last lines before the first repeated line and the start of the repeat. A recurring trigger right before entry, such as a rejection ("No.") followed by a restated hypothesis, or a "Let's re-check" restart, would point to a specific circuit to repair. Entries that look like plain continued computation would leave the copy machinery as the only handle.

```python
# What happens at loop entry under x5? excerpts: ~260 chars before the onset | ~140 chars after (first repeated material).
rng = np.random.default_rng(1)
ex8 = [(r, x) for r, x in zip(cut8, inf8) if x[1] is not None]
for j in rng.choice(len(ex8), 10, replace=False):
    r, (n, p, on) = ex8[j]; ids = tok.encode(S(r)['text']).ids
    # earliest start of the repeated block: first copy begins at on-p
    before = tok.decode(ids[max(0, on-p-70):on-p]); first = tok.decode(ids[on-p:on]); 
    print(f"{r['id'][:26]:26s} p={p:4d} on={on:5d} || ...{asc(before)[-200:]}  <<CYCLE>> {asc(first)[:150]}")
```