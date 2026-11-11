The helper sources are gone because they were defined in executed cells, but the word lists confirm the earlier framing. COMMIT holds wrap-up tokens such as "So", "Therefore", "Final" and "Done"; DOUBT holds second-guessing tokens such as "Wait", "But", "Hmm" and "Actually". The analysis therefore studied the choice between committing and doubting. The generation files store per-problem records with a nested "samples" list, so my accuracy parser read the wrong field. I will look at the sample fields and recompute per-pass accuracy, cut-off rate and length for gen_2 (original), gen_7 (likely the 4-head ablation) and gen_11 (brake x2). That will tell me what the head ablation achieved compared with the neuron edit.

```python
# Hypothesis: recompute per-pass stats from nested samples to reconstruct passes 4 (gen_7) and 7 (gen_11) vs the original gen_2.
def load_gen(fn): return [json.loads(l) for l in open(fn)]
g2r, g7r, g11r = load_gen('/work/gen_2.jsonl'), load_gen('/work/gen_7.jsonl'), load_gen('/work/gen_11.jsonl')
s0 = g2r[0]['samples'][0]; print('sample keys', list(s0.keys()))
def stats(rs):
    S_ = [s for r in rs for s in r['samples']]
    ck = 'correct' if 'correct' in S_[0] else [k for k in S_[0] if 'correct' in k or 'grade' in k][0]
    nt = np.array([s.get('n_tokens', s.get('ntok', 0)) for s in S_])
    cut = np.array([bool(s.get('cut', s.get('truncated', False))) for s in S_])
    acc = np.array([bool(s[ck]) for s in S_])
    return dict(n=len(S_), acc=acc.mean().round(3), cut=cut.mean().round(3), ntok_med=int(np.median(nt)), acc_fin=acc[~cut].mean().round(3) if (~cut).any() else None, nfin=int((~cut).sum()))
for nm, rs in [('gen_2 orig', g2r), ('gen_7', g7r), ('gen_11', g11r)]:
    print(nm, stats(rs), 'ids same as gen_2:', [r['id'] for r in rs] == [r['id'] for r in g2r])
```