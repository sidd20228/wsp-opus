This pins down where the format decision happens. Without the gate (gen_8), the final \boxed is always the first token of its own line (262 of 264). With the gate (gen_12), the final line varies more ("$", "$$", "Final Answer:", "Answer:"), and sometimes \boxed is not written at all. The decision to write \boxed is made at an answer-section line start, which is exactly where the gate closes the heads. Before spending a forward pass, I'll check whether the earlier teacher-forced passes (fwd_6, fwd_11) already contain answer sections with hidden states at the FA-input layers. If they do, I can test whether the junction direction fires at the final-answer line start and whether answer-section positions are linearly separable from thinking line starts.

```python
# Do saved TF passes contain answer sections (post-</think>) with FA-input hidden states? Inspect meta of fwd_6 / fwd_11.
for d_ in ['/work/fwd_6','/work/fwd_11','/work/fwd_2','/work/fwd_4']:
    if not os.path.exists(d_): continue
    meta=json.load(open(f'{d_}/meta.json')); fl=sorted(os.listdir(d_))
    print(d_, [f for f in fl][:14], {k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in meta.items()})
print(type(TF), (list(TF.keys())[:10] if isinstance(TF,dict) else len(TF)), type(TFmeta))
```