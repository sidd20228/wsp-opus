The kernel already holds per-sample loop statistics for gen_11 (period, onset, length of the final repeat run) in L11, and in L2 for the original. I will decode the repeating unit of each looped sample and compare the two models on two measures. The first is whether the loop unit contains a doubt word, which would mean the cycle is still re-entered through "Wait/But/Hmm/Actually". The second, for GSM8K, is whether the reference answer was already written before the loop began. Under the brake, a high share of doubt-containing loop units would point to strengthening the brake further. A sharp drop would mean the loops that survive the edit run on a different engine, such as re-stating computations, and need separate diagnosis.

```python
# Hypothesis: under brake x2 the surviving loops are less often doubt-driven; check doubt words in loop unit + answer-before-onset, orig vs x2.
DW = re.compile(r'\b(Wait|wait|But|Hmm|Actually|However|Alternatively)\b')
def loop_units(G_, Ldf):
    rows = []
    for _, r in Ldf[Ldf.cut & Ldf.looped].iterrows():
        rec = next(x for x in G_ if x['id'] == r['id']); s = rec['samples'][int(r['si'])]
        enc_ = tok.encode(s['text']); ids_l = enc_.ids
        unit = tok.decode(ids_l[-int(r['period']):]) if r['period'] > 0 else ''
        on = int(r['onset']); cpos = enc_.offsets[min(on, len(ids_l)-1)][0]
        pre = s['text'][:cpos]; ans = str(rec['answer']).strip()
        has_ans = (re.search(r'(?<![\d.])' + re.escape(ans) + r'(?![\d])', pre) is not None) if rec['source'].startswith('gsm') else None
        rows.append(dict(id=r['id'], si=r['si'], src=rec['source'], period=r['period'], onset=on, doubt=bool(DW.search(unit)), has_ans=has_ans, unit=unit))
    return pd.DataFrame(rows)
LU2, LU11 = loop_units(g2r, L2), loop_units(g11r, L11)
for nm, d in [('orig', LU2), ('x2', LU11)]:
    gs = d[d.src.str.startswith('gsm')]
    print(nm, 'n looped-cut', len(d), 'doubt-in-unit', round(d.doubt.mean(), 3), 'period med', int(d.period.median()), 'onset med', int(d.onset.median()),
          '| GSM n', len(gs), 'answer-before-onset', round(gs.has_ans.mean(), 3))
for u in LU11.sample(5, random_state=0).unit: print('UNIT:', asc_(u)[:200].replace('\n', ' / '))
```