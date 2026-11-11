This first cut sample has no exact periodic tail and almost no repeated 32-grams (rep 0.0005). It ran out of budget while still producing new text, a failure mode different from the verbatim loops. I'll run the same helpers on all 67 cut samples and tabulate the share with an exact loop, the period and onset distribution, and the repetition fraction. Then I'll compare with gen_9's cut samples. If many cuts are non-repeating long reasoning, the remaining cut problem is either prolonged search on hard problems or near-loops (paraphrased repetition) that exact matching misses.

```python
# Classify gen_12 cut samples: exact loop (period/onset) vs non-periodic long reasoning; compare with gen_9 cuts.
def cut_summary(cuts):
    res=[]
    for (pid,i,s) in cuts:
        L,per_,rep_ = loopinfo(s['text'])
        _, _, on = onset_info(s['text'])
        res.append((pid, i, per_, on, rep_))
    per_l = [r[2] for r in res if r[2] is not None]
    print(f"n={len(res)} exact-loop={len(per_l)} ({100*len(per_l)/len(res):.0f}%) period med={np.median(per_l) if per_l else None}",
          "onset med", np.median([r[3] for r in res if r[3] is not None]) if per_l else None,
          "rep frac quartiles", np.round(np.percentile([r[4] for r in res],[25,50,75]),3))
    return res
cut9 = [(r['id'],i,s) for r in g9 for i,s in enumerate(r['samples']) if s['cut']]
R12 = cut_summary(cut12); R9 = cut_summary(cut9)
print('gen12 non-loop cut examples:', [(r[0],round(r[4],3)) for r in R12 if r[2] is None][:12])
```