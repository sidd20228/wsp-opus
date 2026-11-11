No baseline pass was ever run on dense_v1. It holds 100 GSM8K and 100 MATH reference solutions (short, median about 240 tokens), 24 finished and 24 cut texts from the original model, and the two loop texts, about 224k tokens in all. For the next experiments I need an instrument that matches the current failure mode. It should contain restatement-loop texts from the brake-x2 model (to measure whether an edit lowers loop copy confidence), finished brake-x2 texts (to measure side effects on the model's normal reasoning), and the reference solutions (to check that math knowledge is not damaged). One baseline pass with the current best edit plus one pass per candidate then gives dense paired measurements on identical texts. To build it correctly, I need the exact raw-text format of the evaluation prompt and of the generated samples, so the teacher-forced tokens match what the model produced.

```python
# Hypothesis: (bookkeeping) confirm raw chat formatting so TF texts reproduce generation exactly (prompt + '<think>\n' + generated text).
print(asc_(open('/work/eval_prompt_format.txt').read()[:600]))
fin_ex = next(d for d in dense if d['kind'] == 'fin'); t_ = fin_ex['text']
k_ = t_.find('<|im_start|>assistant'); print('fin text assistant part starts:', repr(asc_(t_[k_:k_+80])))
rec = next(x for x in g11r if x['id'] == fin_ex['id']) if any(x['id'] == fin_ex['id'] for x in g11r) else None
s_ = g11r[5]['samples'][0]['text']; print('gen sample text starts:', repr(asc_(s_[:80])), '| ends:', repr(asc_(s_[-60:])))
print('prompt_text example:', repr(asc_(prompt_text('PROBLEM')[-160:])))
```