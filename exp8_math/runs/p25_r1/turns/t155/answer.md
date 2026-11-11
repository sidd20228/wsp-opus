The drift after a correct answer is consistent: the model re-questions its reading of the problem. Examples are "Is there any chance 'Two of her walls' means...", "Maybe the '4 days later' refers to the upload?", "Did I interpret ... correctly?", "Identify Potential Ambiguities and Interpretations" and "Which is more standard?". This is not "Wait" doubt, which the brake removed; it is ambiguity-hunting on problems that are not ambiguous. Before deciding whether that habit has an editable source, I'll measure how often each kind of line opener occurs in correct versus wrong finished runs, per 1,000 lines, by source. Openers like "Maybe", "What if", "Is there", "Alternatively", "However", "Does", "Perhaps" are much more frequent in wrong runs if this is the drift engine. If wrong and correct runs use them at similar rates, ambiguity-hunting is a symptom of hard problems and not a separate cause.

```python
# Line-opener rates (per 1000 thinking lines) in gate7 finished runs: ok vs wrong, by source. Is ambiguity-hunting a distinct signature?
OPS = {'maybe':r'maybe', 'what if':r'what if', 'is there':r'is there|is it possible|is there any', 'alternatively':r'alternatively',
       'however/but':r'however|but\b', 'does/did/do':r'(?:does|did|do)\b', 'perhaps/could':r'perhaps|could\b', 'let me re/reread':r"let(?:'s| me) (?:re|look at the|check the (?:problem|wording|text))",
       'interpret*':r'.*interpret', 'double check':r'\**\s*double[- ]check|\**\s*sanity|\**\s*self-correction', 'so/therefore/thus':r'so\b|therefore|thus', 'final':r'\**\s*final'}
def opener_rates(keys):
    cnt=collections.Counter(); nl=0
    for k in keys:
        th=byid[k[0]]['samples'][k[1]]['text'].split('</think>')[0]
        for ln in th.split('\n'):
            s=ln.strip().lstrip('*-#> 0123456789.').strip().lower()
            if not s: continue
            nl+=1
            for nm,pt in OPS.items():
                if re.match(pt, s): cnt[nm]+=1
    return {nm: round(1000*cnt[nm]/nl,1) for nm in OPS}, nl
for src in ['gsm8k','math']:
    for st in ['ok','wrong']:
        ks=[k for k,v in T12.items() if v['src']==src and v['st']==st]; r,nl=opener_rates(ks); print(f"{src:5s} {st:5s} n={len(ks)} lines={nl}", r)
```