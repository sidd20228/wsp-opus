The commitment check is clear:

- **Finished answers are long.** The median is 4,500 tokens and the 10th percentile is 1,568, even though 23 of the 48 finished answers are GSM8K.
- **Cut-off answers usually already contain the right number.** In 74% of the 147 cut-off GSM8K samples, the reference number appears in the thinking before the loop starts. Its first appearance has a median of token 678, while the loop starts at a median of token 3,074.

So the model typically has the right number within about 700 tokens, keeps going for a few thousand more tokens without committing, and only then falls into a verbatim loop. That points to H2: a failure to commit and close the thinking. Loops look like the end state of never stopping, so weakening the induction heads alone may just turn verbatim loops into endless non-verbatim re-checking.

Before designing a measurement, I'll read what the model actually writes right after first stating the correct answer. I'll also look at what precedes `</think>` in finished traces. This will show what "deciding to stop" looks like in this model, and what the competing continuation is (for example "Wait", re-quoting the problem, or considering alternative readings). One caveat: a number match can be a coincidence, so I'll also check how often the number already appears in the problem itself.

```python
# Look at what the model writes right after first stating the correct answer (cut GSM8K), and what precedes </think> in finished traces.
import random
random.seed(0)
probs_by_id = {json.loads(l)['id']: json.loads(l) for l in open('/practice/gsm8k_train.jsonl')}
cut_g = D3[(D3.src == 'gsm8k') & (D3.cut)]
in_prob = 0; examples = []
for _, r_ in cut_g.iterrows():
    v = num_norm(r_.ans); sv = str(int(v)) if v is not None and v == int(v) else str(r_.ans)
    pat = r'(?<![\d.])' + re.escape(sv) + r'(?![\d])'
    if re.search(pat, probs_by_id[r_.id]['problem']): in_prob += 1; continue
    m_ = re.search(pat, r_.txt[:int(len(r_.txt) * 0.3)])
    if m_: examples.append((r_.id, r_.si, sv, m_.start(), r_.txt))
print('ref number already in problem text:', in_prob, 'of', len(cut_g), '; usable examples', len(examples))
def asc(s): return s.encode('ascii', 'replace').decode().replace('\n', ' | ')
for ex_ in random.sample(examples, 4):
    pid_, si_, sv, st, txt = ex_
    print('==', pid_, si_, 'ans', sv, '| after first mention:', asc(txt[st - 80:st + 220]))
fin_g = D3[(~D3.cut)]
for _, r_ in fin_g.sample(3, random_state=1).iterrows():
    k_ = r_.txt.find('</think>')
    print('-- finished', r_.id, 'ntok', r_.ntok, '| before </think>:', asc(r_.txt[max(0, k_ - 260):k_ + 10]))
```