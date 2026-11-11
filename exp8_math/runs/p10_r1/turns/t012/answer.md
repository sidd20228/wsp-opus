Loops start around token 4,400 (median; interquartile range about 2,500–6,700) and have very short periods: 173 of 435 repeat a single line, 87 alternate two lines, and almost all repeat within 8 lines. That is the classic self-reinforcing copy loop. Before looking for the mechanism I want to see what is repeated, using short excerpts from about a dozen cut answers. If the looped lines are generic verification fillers ("Yes.", "Wait, ...", restating the same equation), the loop is driven by copying from context rather than by reasoning.

```python
# Show the looped unit (last few lines) for a sample of cut answers, truncated
rs = np.random.default_rng(1)
cut_samples=[(r['id'],s['text']) for r in rows for s in r['samples'] if s['cut']]
for idx in rs.choice(len(cut_samples), 12, replace=False):
    i,t=cut_samples[idx]
    lines=[l for l in t.split('\n') if l.strip()]
    unit=' | '.join(l.strip()[:90] for l in lines[-3:])
    print(i[:35], '::', unit.encode('ascii','replace').decode()[:280])
```