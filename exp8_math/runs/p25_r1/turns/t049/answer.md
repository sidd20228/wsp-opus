Both measures support a progressive doubt ramp in the stuck samples. Their "Wait" density is already twice that of finished samples in the first 1k tokens (2.7 vs 1.2 per 1k) and keeps rising (6.0 at 2-3k, about 9.5 later; finished samples go from 1.2 to about 4.5). Before any exact loop, 9.3% of their long lines repeat an earlier line, against 2.9% for finished samples. So the failure develops as an escalating re-verification mode, with re-stated lines, long before the verbatim cycle locks in. This fits E1's weak effect: removing copying heads only changes the end state.

My working hypothesis is that a "reflection / re-check" signal in the residual stream decides, at each line start, whether the next line opens a new doubt ("Wait, is it possible...") or continues or concludes. If I can find it as a direction, I can test a directional ablation in the writing layers' output matrices. I will first recover the prompt format and the tokenization of a typical "Wait" line start, so I can locate the exact decision positions in teacher-forced texts.

```python
# Setup for direction-finding TF pass: recall prompt format, splits, and how a 'Wait' line start tokenizes (to locate decision positions).
print(open('/work/eval_prompt_format.txt').read()[:600])
print({k: len(v) for k, v in sp.items()})
for s_ in ['\n    *   Wait, is', '\n\n    Wait, I', '\n        *   Wait,', '.\n    *   So, 300.', '\n    *   Let\'s check']:
    ids_ = tok.encode(s_).ids; print(repr(s_), ids_, [repr(tok.decode([i])) for i in ids_])
tfs = json.load(open('/work/tf_small.json')); print(type(tfs), len(tfs), [len(t) for t in tfs] if isinstance(tfs, list) else list(tfs.keys()))
```