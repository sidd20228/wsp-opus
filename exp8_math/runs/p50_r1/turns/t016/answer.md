Now the picture is quantified. 98.5% of the 464 cut generations end in an exact loop (≥95% of 32-grams repeated from the onset to the end). The loop starts at a median of ~4,100 tokens (10th–90th percentile: 1,300–10,000) and has a short period (median 43 tokens, i.e. one or two lines). 84% of problems loop in both samples, and finished answers (median 4,500 tokens) are essentially always correct. So the model can usually solve the problem; it just falls into a self-copying attractor and never escapes. Next, I want to characterize the loop units: their content (does the repeated line start with "Wait"?), their indentation depth (the example was deeply nested), and what happens just before onset. A common trigger, such as deep bullet nesting, would point to a specific mechanism; diverse loop content would point to a general copy circuit.

```python
# Hypothesis: loops have a stereotyped form (deeply indented bullet lines, often "Wait"); characterize loop units.
import re
ex = []
for r in df[df.cut & df.onset.notna()].itertuples():
    ids = IDS[(r.id, r.si)]
    on = int(r.onset); per = int(r.period) if r.period else 40
    unit = tok.decode(ids[on + 2*per: on + 3*per])
    pre = tok.decode(ids[max(0, on-60): on])
    ind = [len(l) - len(l.lstrip(' ')) for l in unit.split('\n') if l.strip()]
    ex.append(dict(id=r.id, unit=unit, pre=pre, indent=max(ind) if ind else 0, wait=('Wait' in unit), nl=unit.count('\n')))
ed = pd.DataFrame(ex)
print('share of loop units containing "Wait":', ed.wait.mean(), ' indent quantiles:', ed.indent.quantile([.1,.25,.5,.75,.9]).tolist())
print('newlines per unit quantiles:', ed.nl.quantile([.1,.5,.9]).tolist())
for i in range(0, 60, 6):
    print(f'[{i}] indent={ed.indent[i]} UNIT:', asc(repr(ed.unit[i][:160])))
```