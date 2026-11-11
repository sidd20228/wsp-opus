The ablation's cost is mostly where it does no harm. Flips concentrate at positions that were already uncertain:
- margin under 1: 17–29% flip;
- margin 2–4: 2–4% flip;
- margin above 4: 0.2–1.6% flip.

The one real cost is copying numbers. In text 2, the top choice flips at 12% of digit positions, and the share of confidently copied digits drops from 76% to 58%. Some flips are formatting ("6666" vs "6,666"), but some are genuine digit changes. Full ablation is therefore too strong to ship as is. A partial attenuation should roughly halve both the loop-breaking effect and this cost.

Before spending a generation pass, I want to estimate how much accuracy breaking loops could buy. If cut-off answers often already contain the correct answer before they lock into a loop, freeing the model from loops should convert directly into correct finished answers. If they rarely contain it, loops are more a symptom of being lost, and a copy-damping edit would mostly produce other, non-verbatim failures. I'll check pass 1's cut answers: whether the reference answer appears (boxed, or stated as the result) and whether it appears before the dense loop starts.

```python
# Upside estimate: do cut answers already contain the correct answer before the loop starts? (if often, breaking loops should convert to accuracy)
import re
def norm_ans(s): return re.sub(r'\s+|\\!|\\,|\$|\\left|\\right|\\dfrac|\\tfrac', lambda m: '\\frac' if 'frac' in m.group(0) else '', str(s)).strip('.')
def boxed_all(t):
    out, i = [], 0
    while True:
        j = t.find('\\boxed{', i)
        if j < 0: return out
        k, dep = j + 7, 1
        while k < len(t) and dep:
            dep += {'{': 1, '}': -1}.get(t[k], 0); k += 1
        out.append(t[j + 7:k - 1]); i = k
def loop_onset_char(t, win=200):
    lines = t.split('\n'); seen = set(); pos = 0; run = 0
    for ln in lines:
        s = ln.strip()
        if s and s in seen: run += 1
        else: run = 0
        if s: seen.add(s)
        if run >= 6: return pos
        pos += len(ln) + 1
    return None
stats = dict(n=0, boxed_any=0, boxed_correct=0, ans_stated=0, ans_before_loop=0, loop_found=0)
for r in g1:
    ref = norm_ans(r['answer'])
    for s in r['samples']:
        if not s['cut']: continue
        t = s['text']; stats['n'] += 1
        bx = [norm_ans(b) for b in boxed_all(t)]
        stats['boxed_any'] += bool(bx); stats['boxed_correct'] += any(b == ref for b in bx)
        on = loop_onset_char(t); stats['loop_found'] += on is not None
        pat = re.compile(r'(?:answer is|=|is)\s*\$?\\?(?:boxed\{)?' + re.escape(str(r['answer']).strip()) + r'(?![0-9])')
        m = pat.search(t)
        stats['ans_stated'] += m is not None
        if m is not None and on is not None and m.start() < on: stats['ans_before_loop'] += 1
print(stats)
```