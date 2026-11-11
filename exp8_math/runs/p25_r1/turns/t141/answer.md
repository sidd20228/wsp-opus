The regex comparison is uninformative: it gives about 55% "mismatch" in every configuration because it catches LaTeX fragments. It shows no gate-specific copy damage, and I'll drop it rather than refine it. The evidence for the answer section points only to the missing \boxed. Next I'll prepare the teacher-forced pass that collects hidden states at the four gated layers' inputs (after layers 10, 14, 18, 22) over full answer sections, under the current edit. For that I need the exact raw-text format of the evaluation prompt plus generated response, as used in earlier TF passes. I'll check the saved prompt format, the build helper, and how sample texts begin.

```python
# Recover the raw-text format for TF passes (eval prompt + generated text) and check the build() helper.
fmt = open('/work/eval_prompt_format.txt').read(); print(repr(fmt[:400]))
print(inspect.signature(build)); s=g12[0]['samples'][0]['text']; print(repr(s[:80]))
try:
    b = build(g12[0]['id'], s) if len(inspect.signature(build).parameters)==2 else None; print(repr(b[:300]) if b else None, repr(b[-60:]) if b else None)
except Exception as e: print('build err', e)
```