`gen_2` stores 256 problems with 2 samples each. `df` already holds per-sample loop onset, period and length. Next I'll do the H1/H2 check. For each cut-off GSM8K sample, I'll ask whether the reference number was already stated before the loop began. If it was, roughly where is that point, and what does the model write right after it? I'll also look at the lengths of the 48 finished answers. If most cut samples contain the right answer early, then committing and finishing is a bigger problem than copying itself.

```python
# H2 check: does the reference answer appear in the thinking before loop onset in cut GSM8K samples? Lengths of finished answers?
s0_ = G2[0]['samples'][0]; print(list(s0_.keys()))
def num_norm(a):
    a = str(a).replace(',', '').strip()
    try: return float(a)
    except: return None
rows3 = []
for g_ in G2:
    for si_, smp in enumerate(g_['samples']):
        r_ = df[(df.id == g_['id']) & (df.si == si_)].iloc[0]
        txt = smp.get('text') or smp.get('completion') or smp.get('output')
        rows3.append(dict(id=g_['id'], src=g_['source'], si=si_, ans=g_['answer'], txt=txt, cut=r_.cut, correct=r_.correct, ntok=r_.ntok, onset=r_.onset))
D3 = pd.DataFrame(rows3)
fin = D3[~D3.cut]
print('finished n=%d, ntok quantiles' % len(fin), np.percentile(fin.ntok, [10, 25, 50, 75, 90]).astype(int), 'src counts', fin.src.value_counts().to_dict())
# GSM8K cut: first occurrence of the reference number before onset
hits = []; first_tok = []
for _, r_ in D3[(D3.src == 'gsm8k') & (D3.cut)].iterrows():
    v = num_norm(r_.ans); enc_ = tok.encode(r_.txt); offs = enc_.offsets
    on_tok = int(r_.onset) if not np.isnan(r_.onset) else len(offs)
    on_char = offs[min(on_tok, len(offs) - 1)][0]
    pre_txt = r_.txt[:on_char]
    pat = r'(?<![\d.])' + re.escape(str(int(v)) if v is not None and v == int(v) else str(r_.ans)) + r'(?![\d])'
    m_ = re.search(pat, pre_txt)
    hits.append(m_ is not None)
    if m_: first_tok.append(sum(1 for o in offs if o[0] < m_.start()))
print('GSM8K cut samples:', len(hits), 'ref answer appears before onset in %.1f%%' % (100 * np.mean(hits)))
print('first-appearance token quantiles', np.percentile(first_tok, [10, 25, 50, 75, 90]).astype(int) if first_tok else None)
print('onset quantiles (gsm cut)', np.nanpercentile(D3[(D3.src == 'gsm8k') & (D3.cut)].onset, [10, 25, 50, 75, 90]).astype(int))
```