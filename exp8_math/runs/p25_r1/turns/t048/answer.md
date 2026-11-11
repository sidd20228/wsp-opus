The saved data are float32: hidden states at all 25 layers cost about 100 KB per token, and the MLP activations about 14 KB per layer per token. The 6 GB limit therefore allows roughly 55k tokens of hidden states, or about 25k with eight MLP layers as well. Before spending a pass, I want to know whether the stuck samples show a progressive rise in self-doubt and in re-stated lines before the exact loop. That would mean a "doubt / re-verify" process drives the descent and the exact loop is its end point, which argues for targeting the doubt signal rather than only the copying heads. Concretely, I will compare cut against finished samples on "Wait" density in successive 1k-token windows and on the share of long lines that duplicate an earlier line. Rising curves before onset in cut samples, and flat low ones in finished samples, would support that picture.

```python
# Hypothesis: before exact-loop onset, cut samples show rising 'Wait' density and rising duplicate-line rate (semantic looping), unlike finished ones.
g1 = readj('/work/gen_1.jsonl')
def profile(text, upto=None):
    ids = ids_of(text); n = len(ids) if upto is None else upto
    out = []
    for w0 in range(0, min(n, 12000), 1000):
        seg = tok.decode(ids[w0:min(n, w0+1000)].tolist())
        out.append(len(re.findall(r'\bWait\b', seg)))
    lines = [l.strip() for l in tok.decode(ids[:n].tolist()).split('\n') if len(l.strip()) > 40]
    seen, dup = set(), 0
    for l in lines:
        dup += l in seen; seen.add(l)
    return out, dup / max(1, len(lines)), len(lines)
rows = []
for recs, tag in [(g1, 'p1'), (g5o, 'p4')]:
    for r in recs:
        for j, s in enumerate(r['samples']):
            ids_n = None
            if s['cut']:
                p, on = tail_period(ids_of(s['text']))
                if on is None: continue
                ids_n = on
            w, dupr, nl = profile(s['text'], ids_n)
            rows.append(dict(cut=s['cut'], w=w, dup=dupr, nl=nl, ok=s['correct']))
pf = pd.DataFrame(rows)
for c in [False, True]:
    sub = pf[pf.cut == c]; W = np.full((len(sub), 12), np.nan)
    for i, w in enumerate(sub.w): W[i, :len(w)] = w
    print('cut' if c else 'fin', len(sub), 'Wait per 1k by window:', np.nanmean(W, 0).round(1)[:10].tolist(), 'n per win', (~np.isnan(W)).sum(0)[:10].tolist())
    print('   dup-line rate mean %.3f median %.3f' % (sub.dup.mean(), sub.dup.median()))
```