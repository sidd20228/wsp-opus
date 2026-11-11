The paired result on setA is decisive. Accuracy rises from 7.8% to 44.5%, a paired gain of +36.7 ± 3.1 points (96 problems wrong → right, only 2 right → wrong). It holds for both sources: GSM8K 10.9% → 49.2%, MATH 4.7% → 39.8%. The cut share falls from 93% to 52%. The model still thinks normally when it finishes: median 4.9k tokens, against 5.6k for the original's finished answers, so it is not cutting reasoning short, and 89.6% of finished answers are correct. setB, never used before, gives 42.6% with 54% cut. The two runs share the first ~90 tokens (median) before diverging, so the seed pairing holds.

So strengthening the native "Wait" brake (one column of one matrix) repairs most of the non-termination. I will save this edit as best_1 and record it. Then I will verify the mechanism in the generated texts: "Wait" density, use of substitute doubt openers, duplicate-line rate, loop share and onset, compared with the original on setA. That also shows what the remaining 52% of cut answers look like.

```python
# Save E2 as best_1; mechanism check in generated text (E2 vs orig on setA): Wait density, doubt openers, dup lines, loops.
import shutil
if os.path.exists('/work/best_1'): shutil.rmtree('/work/best_1')
shutil.copytree('/work/edited', '/work/best_1'); print(os.listdir('/work/best_1'))
def text_stats(recs, ids_keep):
    rows = []
    for r in recs:
        if r['id'] not in ids_keep: continue
        s = r['samples'][0]; t = s['text']; t_think = t[:t.find('</think>')] if '</think>' in t else t
        ids = ids_of(t_think); p, on = tail_period(ids) if s['cut'] else (None, None)
        pre = tok.decode(ids[:on].tolist()) if on is not None else t_think; ntok = on if on is not None else len(ids)
        lines = [l for l in pre.split('\n') if len(l.strip()) > 3]; fw = [m_.group(1) for l in lines if (m_ := fw_re.match(l))]
        c_ = Counter(fw); nl = max(1, len(fw))
        long_ = [l.strip() for l in lines if len(l.strip()) > 40]; seen = set(); dup = 0
        for l in long_: dup += l in seen; seen.add(l)
        rows.append(dict(cut=s['cut'], loop=on is not None, onset=on, wait=1000 * c_['Wait'] / nl, other_doubt=1000 * sum(c_[w] for w in ['Is', 'Could', 'Maybe', 'But', 'Hmm', 'Hold', 'Actually', 'What']) / nl,
                         lets=1000 * c_["Let's"] / nl, commit=1000 * sum(c_[w] for w in ['Final', 'Okay', 'Done', 'I', 'Step']) / nl, dup=dup / max(1, len(long_))))
    return pd.DataFrame(rows)
tsO = text_stats(g5o, setA_ids); tsE = text_stats(g7, setA_ids)
for nm, d in [('orig', tsO), ('E2', tsE)]:
    print(nm, 'per 1000 lines: Wait %.0f, other doubt %.0f, Let\'s %.0f, commit %.0f | dup-line %.3f | exact-loop share %.2f, onset median %s' % (
        d.wait.mean(), d.other_doubt.mean(), d.lets.mean(), d.commit.mean(), d.dup.mean(), d.loop.mean(), d.onset.median()))
```