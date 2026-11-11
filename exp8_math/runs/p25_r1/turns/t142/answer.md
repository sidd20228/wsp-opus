The regex result rules out answer-copying damage, but writing the plan down exposes a problem with the answer-section exemption. Any linear change to the gate logit that releases answer-section line starts also shifts every other answer-section position. That would open the gates there, roughly four-fold, unless the model has a feature specific to "answer line starts". The same applies if the needed signal is only "line start" plus "in the answer" combined additively. A head-level exemption is more promising: un-gate whichever tier-2 head (L19H5, L23H0, L23H4, L23H7) carries the format, since E1 without them produced no format failures. First I'll check a cheap clue about the mechanism: whether the thinking itself usually contains a \boxed{...} that induction could copy into the final line, and whether that differs between configurations and in the no-box samples. If gen_12 thinking contains \boxed less often, the format loss could be partly upstream of the answer line.

```python
# Is there a \boxed in the thinking to copy from? Share of finished samples with \boxed inside <think>, by config and for gen_12 noans.
def think_box(path, filt=None):
    n=0; k=0
    for r in readj(path):
        for i,s in enumerate(r['samples']):
            if s['cut'] or '</think>' not in s['text']: continue
            if filt and not filt(r['id'],i): continue
            n+=1; k += ('\\boxed' in s['text'].split('</think>')[0])
    return n,k
for nm in ['gen_7','gen_8','gen_9','gen_10','gen_12']:
    n,k=think_box(f'/work/{nm}.jsonl'); print(f"{nm}: finished {n}, thinking contains \\boxed: {k} ({100*k/n:.0f}%)")
nb_keys=set(zip(na.id,na.i)); n,k=think_box('/work/gen_12.jsonl', lambda pid,i:(pid,i) in nb_keys); print('gen_12 noans:', n, k)
toks_prompt = tok.encode('put your final answer within \\boxed{}.<|im_end|>').ids; print('prompt tail ids', toks_prompt)
```