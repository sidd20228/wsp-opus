97% of cut samples end in an exactly periodic verbatim cycle, typically 18-172 tokens long (median 48). This is textbook repetition-loop degeneration: once a span recurs, the model copies its own earlier text, which points to an induction-style copy circuit. Before localizing, I want to know what the loops contain. If they are mostly "So the answer is X. Wait, let me double check..." cycles, the model has the answer and fails to wrap up. If they are re-derivations or enumerations, it is losing track mid-reasoning. I'll print one cycle (≤300 chars) from 12 random looping samples, plus the ~150 characters just before each loop onset.

```python
# Hypothesis: loop cycles are mostly self-doubt/verification text ("Wait, ...") restating content; inspect cycles.
rs = np.random.default_rng(0)
cands = R1[(R1.cut)&(R1.period.notna())&(R1.onset.notna())].index.tolist()
for i in rs.choice(cands, 12, replace=False):
    ids=TOK[i]; p=int(R1.period[i]); on=int(R1.onset[i])
    cyc = tok.decode(ids[-p:]); pre = tok.decode(ids[max(0,on-60):on])
    print(f'--- {R1.id[i]} onset={on} p={p} correct_ans={ans_map[R1.id[i]][:20]}')
    print('  PRE :', asc(pre)[-150:])
    print('  CYC :', asc(cyc)[:300])
```