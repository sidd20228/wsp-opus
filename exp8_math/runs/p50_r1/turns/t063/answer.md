Neuron 998 in layer 16 looks like a natural "Wait-writer". It is not a weight anomaly: its gate, up and down norms are all at the layer median (z ≈ 0).

- **Output.** Its output vector is almost exclusively the wait family: ' wait', 'wait', 'Wait', ' Wait', ' WAIT', '_wait' and '(wait' are the most extreme tokens in the whole vocabulary, at z of −11 to −18.
- **When it fires.** It fires negatively, which promotes those tokens. That happens on about 10% of all tokens in these traces (q10 = −1.9), with a median of −2.8 at "Wait" positions. Its strongest firing comes right before a newline, i.e. at sentence ends where the next line's opening is being decided.
- **Size of effect.** By direct attribution it adds about +3 nats to "Wait" over the alternatives at every such decision point. That is enough to turn a mild preference against "Wait" into a habit.

To test edits cheaply and repeatably, I'll build a fixed dense text set and run the original model on it once. Every later edit then needs only one teacher-forced pass for a paired, per-token comparison. The set will contain:

- texts A and B;
- about 24 finished, correct original traces, including their wrap-up and `</think>`;
- about 24 cut traces up to just past loop onset;
- 200 human-written GSM8K/MATH reference solutions as a capability check that is not biased toward the original's own samples.

This run builds and saves the set and reports its token count. I'll also drop the four-head ablation, which was rejected.

```python
# Build a reusable dense TF text set (own traces: finished/cut-to-onset; texts A,B; 200 human reference solutions) and drop rejected edit.
for nm_ in list(edited()): drop_edit(nm_)
print('edits now:', edited())
AB = json.load(open('/work/attn_texts_AB.json')); print(type(AB), (list(AB.keys()) if isinstance(AB, dict) else len(AB)))
def prompt_text(problem):
    return '<|im_start|>user\n' + problem + '\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'
probtext = {}
for fn_ in ['/practice/gsm8k_train.jsonl', '/practice/math_train.jsonl']:
    for l_ in open(fn_):
        o_ = json.loads(l_); probtext[o_['id']] = o_
rng2 = np.random.RandomState(7)
dense = []   # dicts: kind, id, si, text
fin_all = L2[~L2.cut].merge(D3[['id', 'si', 'txt']], on=['id', 'si'])
fin_all = fin_all[fin_all.n <= 6000]
for _, r_ in fin_all.sample(min(24, len(fin_all)), random_state=1).iterrows():
    dense.append(dict(kind='fin', id=r_.id, si=int(r_.si), text=prompt_text(probtext[r_.id]['problem']) + r_.txt))
cut_all = L2[(L2.cut) & (L2.looped) & (L2.onset > 1500) & (L2.onset < 5000)].merge(D3[['id', 'si', 'txt']], on=['id', 'si'])
for _, r_ in cut_all.sample(24, random_state=2).iterrows():
    offs = tok.encode(r_.txt).offsets; endc = offs[min(int(r_.onset) + 400, len(offs) - 1)][0]
    dense.append(dict(kind='cut', id=r_.id, si=int(r_.si), onset=int(r_.onset), text=prompt_text(probtext[r_.id]['problem']) + r_.txt[:endc]))
used = set(sum([splits[k] for k in splits], []))
gs_ref = [o_ for k_, o_ in probtext.items() if k_.startswith('gsm8k') and k_ not in used]
ma_ref = [o_ for k_, o_ in probtext.items() if k_.startswith('math') and k_ not in used and len(o_.get('solution', '')) < 1500]
for o_ in list(rng2.choice(gs_ref, 100, replace=False)) + list(rng2.choice(ma_ref, 100, replace=False)):
    sol = re.sub(r'<<[^>]*>>', '', o_['solution'])
    dense.append(dict(kind='ref_' + o_['id'].split('_')[0], id=o_['id'], text=prompt_text(o_['problem']) + sol))
ntoks = [len(tok.encode(d_['text']).ids) for d_ in dense]
print(len(dense), 'texts; total tokens', sum(ntoks), '| by kind', Counter(d_['kind'] for d_ in dense), '| max len', max(ntoks))
json.dump(dense, open('/work/dense_v1.json', 'w'))
```