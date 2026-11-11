Correctly scaled, the logit lens gives a clear and more specific picture. Up to the input of the last layer, the copy margin grows similarly on ordinary text and inside loops (hidden_20: 4.3 pre-loop, 5.3 at cycle 0, 5.4 late). Two jumps come from layers with induction heads (layer 15: +1 to 1.5; layer 19: +1). The final layer (23) then behaves very differently by context: it slightly lowers the margin on ordinary text (5.53 → 4.97) and at the loop's first cycle (5.98 → 5.74), but adds about +5 nats in late cycles (7.80 → 12.75). So the last layer is a loop-specific copy amplifier, consistent with its induction head H5 attending 0.74 in loops versus 0.28 pre-loop. Pass 7 already showed that making loops escapable by ablating copy heads did not help: the model escaped and re-looped, and normal copying suffered. So the root problem is the stuck re-verification state that loops crystallize from, not the copy machinery itself. The brake worked because it acted at that state's native decision point (the doubt token at a line start). I will check whether the surviving restatement loops have a comparably dominant line-start decision, such as re-check openers like "Let's check" or "Let's try". If they do, an analogous native component may control it.

```python
# Hypothesis: surviving x2 restatement loops are dominated by a few line-start openers (e.g. "Let's check/try", re-quoting), i.e. a re-check decision.
from collections import Counter
def line_openers(unit):
    outs = []
    for ln in unit.split('\n')[1:]:
        s = ln.strip().lstrip('*').strip()
        if s: outs.append(' '.join(s.split()[:2])[:20])
    return outs
c11, c2_ = Counter(), Counter()
for u in LU11[~LU11.doubt].unit: c11.update(set(line_openers(u)))
for u in LU2.unit: c2_.update(set(line_openers(u)))
print('x2 non-doubt loops n=', (~LU11.doubt).sum(), 'top openers (units containing):', [(asc_(k), v) for k, v in c11.most_common(18)])
print('orig loops n=', len(LU2), 'top openers:', [(asc_(k), v) for k, v in c2_.most_common(12)])
first_word = Counter(w.split()[0] for u in LU11[~LU11.doubt].unit for w in set(line_openers(u)) if w)
print('x2 first words:', [(asc_(k), v) for k, v in first_word.most_common(15)])
```