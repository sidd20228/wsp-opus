The closing decision depends strongly on where the model is in its outline:

- **Finished texts:** `</think>` gets p = 0.44 after "*Looks good.*" (T3), 0.72 after "Done." (T4) and 0.067 after "I'll structure the response accordingly." (T5). All three closings happen from a 4-space-indented line.
- **T2 (stuck, also at 4-space indent):** the model's own conclusions give `</think>` p = 0.15 after "Okay, I'll proceed." and 0.04 after "Okay, I'll write the response.", but under 0.002 after "Okay, I'm done.". So closing is contested there, and the model keeps choosing the indent that starts "Wait".
- **T0 and T1 (stuck GSM8K):** `</think>` is essentially impossible everywhere (max 0.0006). There the model sits inside step 1 ("Analyze the Request") in sub-bullets indented 8-12 spaces ("            *   Wait, ...").

New hypothesis: the stuck runs are trapped at deep nesting. Each new line copies the previous line's indentation and bullet, the model never dedents back to the outline, and closing only happens from shallow lines. I will test this on all generations. I will compare the indentation of lines over time in cut and finished samples, and the depth at loop onset with the depth at `</think>`. In the teacher-forced texts I will measure the probability that the next line is shallower, the same, or deeper.

```python
# Hypothesis: stuck runs drift to deep indentation and never dedent; closing happens only from shallow (<=4 spaces) lines.
def indents(text):
    return [len(l) - len(l.lstrip(' ')) for l in text.split('\n') if l.strip()]
rows = []
for recs in (g1, g5o):
    for r in recs:
        for s in r['samples']:
            t = s['text']; fin_ = '</think>' in t
            if fin_: body = t[:t.find('</think>')]
            else:
                ids = ids_of(t); p, on = tail_period(ids)
                if on is None: continue
                body = tok.decode(ids[:on].tolist())
            ind = indents(body)
            if len(ind) < 10: continue
            q = np.array(ind); thirds = np.array_split(q, 3)
            rows.append(dict(fin=fin_, last=q[-5:].mean(), deep8=(q >= 8).mean(), t1=thirds[0].mean(), t2=thirds[1].mean(), t3=thirds[2].mean(), maxd=q.max()))
ind_df = pd.DataFrame(rows)
print(ind_df.groupby('fin')[['last', 'deep8', 't1', 't2', 't3', 'maxd']].median().round(2))
print('share with last-5-line indent <=4:', ind_df.groupby('fin').apply(lambda d: (d['last'] <= 4).mean()).round(2).to_dict())
```