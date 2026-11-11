The loops are verification loops nested deep in the model's bulleted outline. Typical cycles: "*   Wait, is it possible that 'twice as many apples as oranges' means A = 2O? Yes. Is it possible...", or "Wait, 3.777... is NOT / IS 3 + 7/99?". Almost always the answer has already been reached ("So, 300.", "38."). The model never returns from the sub-bullets to a top-level step that would wrap up. Copying then locks the cycle in, but the underlying failure is that the model stays in "re-check" mode.

Before I go after mechanisms, I want to see what a successful exit looks like. In finished samples, what precedes `</think>`? Is it a top-level numbered step ("7.  **Final Answer**...")? And how long after first stating the correct answer does the model finish? If finished answers conclude by returning to a top-level outline step soon after verification, the junction to study is the line start where the model picks deeper "*   Wait" over a new top-level step.

```python
# Hypothesis: finished thinking ends via a top-level outline step soon after the answer is found; cut samples never return to top level.
fin = [(r, s) for recs in (g5o, g5e) for r in recs for s in r['samples'] if not s['cut']]
print('n finished', len(fin))
for r, s in fin[:5]:
    t = s['text']; k = t.find('</think>')
    print('---', r['id'], 'tok', s['tokens'], 'correct', s['correct'], '| tail before </think>:', repr(t[max(0, k-250):k][-250:]))
# top-level step headers like "\n\n6.  **Title" ; count & position of last one relative to loop onset in cut samples
hdr = re.compile(r'\n\n?(\d+)\.\s+\*\*([^*\n]{0,40})')
rows = []
for _, row in res['orig'].iterrows():
    s = rec_by_id[row.id]['samples'][0]; t = s['text']; ids = ids_of(t)
    on = int(row.onset) if row.onset == row.onset else len(ids); pre = tok.decode(ids[:on].tolist())
    hs = list(hdr.finditer(pre)); rows.append(dict(nh=len(hs), last_num=int(hs[-1].group(1)) if hs else 0,
        last_title=hs[-1].group(2)[:25] if hs else '', chars_since=len(pre) - hs[-1].end() if hs else len(pre)))
hd = pd.DataFrame(rows); print(hd.nh.describe()[['mean', '50%']].round(1).to_dict(), 'chars since last header med', hd.chars_since.median())
print(hd.last_title.str.strip().value_counts().head(12).to_dict())
```