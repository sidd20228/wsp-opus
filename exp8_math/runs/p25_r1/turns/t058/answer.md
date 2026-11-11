These texts are poor for the decision analysis. Text 0 is almost entirely loop (onset at token 87, period 50), and text 1 has only about 400 pre-loop tokens before 5.5 cycles. Their "Wait" line starts mostly lie inside repeated cycles, and the healthy texts contain none. So they can show how neurons behave across cycles, but not what separates a fresh "Wait" decision from a commitment.

I will build a compact new set. Three stuck samples will each contribute the prompt, the first ~120 answer tokens, the ~450 tokens of verification descent right before loop onset, and 2 cycles. Three finished samples will each contribute the prompt, the first ~120 tokens, and the last ~450 tokens before `</think>` plus the start of the final answer, which captures the commitment lines. All splices fall on line boundaries. This keeps the paste near 15-20k characters while covering three things: doubt line starts, commitment line starts, and loop onset.

```python
# Build compact TF set: 3 stuck (pre-onset descent + 2 cycles) + 3 finished (end of thinking with commitment), spliced at line boundaries.
prob_text = {}
for f in ['/practice/gsm8k_train.jsonl', '/practice/math_train.jsonl']:
    for l in open(f):
        o_ = json.loads(l); prob_text[o_['id']] = o_['problem']
PREFIX = '<|im_start|>user\n'; SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'
def cut_at_nl(ids, a, b):  # decode ids[a:b], starting after first newline and ending at last newline
    s = tok.decode(ids[a:b].tolist()); i = s.find('\n') if a > 0 else -1; j = s.rfind('\n')
    return s[i + 1: j + 1]
cands = res['orig'][(res['orig'].per.between(25, 120)) & (res['orig'].onset.between(1500, 8000)) & res['orig'].found].copy()
cands['plen'] = cands.id.map(lambda i: len(prob_text[i]))
pick_cut = list(cands[cands.src == 'gsm8k'].sort_values('plen').id[:2]) + list(cands[cands.src == 'math'].sort_values('plen').id[:1])
fins = [(r, s) for r in g5o for s in r['samples'] if not s['cut'] and s['correct'] and '</think>' in s['text'] and s['tokens'] > 1500]
fins = sorted(fins, key=lambda x: len(prob_text[x[0]['id']]))[:3]
newset, newmeta = [], []
for i_ in pick_cut:
    s = rec_by_id[i_]['samples'][0]; ids = ids_of(s['text']); row = res['orig'][res['orig'].id == i_].iloc[0]; on, p = int(row.onset), int(row.per)
    body = cut_at_nl(ids, 0, 120) + cut_at_nl(ids, on - 450, on + 2 * p + 1)
    newset.append(PREFIX + prob_text[i_] + SUFFIX + body); newmeta.append(dict(kind='stuck', id=i_, p=p))
for r, s in fins:
    t = s['text']; k_ = t.find('</think>'); ids = ids_of(t[:k_])
    body = cut_at_nl(ids, 0, 120) + cut_at_nl(ids, len(ids) - 450, len(ids)) + t[k_: k_ + 120]
    newset.append(PREFIX + prob_text[r['id']] + SUFFIX + body); newmeta.append(dict(kind='fin', id=r['id']))
print([(m_['kind'], m_['id'], len(ids_of(t))) for m_, t in zip(newmeta, newset)], 'total chars', sum(len(t) for t in newset))
json.dump(dict(texts=newset, meta=newmeta), open('/work/tf_set2.json', 'w'))
```