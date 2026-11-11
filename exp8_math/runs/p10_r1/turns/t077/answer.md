The full removal holds up under scrutiny.

**Accuracy (paired)**

| Comparison | Change | SE | wrong→right | right→wrong |
|---|---|---|---|---|
| E3 vs original | +18.0 points | 2.1 | 112 | 20 |
| E3 vs E2 | +8.8 points | 1.9 | | |

The original-vs-E3 gain holds on both subsets: MATH +16.7 (SE 2.3) and GSM8K +21.9 (SE 4.8).

**Specificity**
- "Wait" use in the first 1k tokens falls only modestly (2.5 → 1.9 per 1,000 tokens).
- The escalation is largely removed: 4–8k tokens 17.7 → 4.5, 8–16k 26.2 → 8.3.
- On the 41 problems both models finish, E3 thinks longer (median 7.7k vs 5.0k tokens), so it is not closing early.

**Costs**
- 32 of E3's 36 wrong finished answers are problems the original never finished.
- Of the 20 problems the original solved and E3 lost, 15 are now cut off rather than answered wrongly. So E3 occasionally still gets stuck, but rarely trades a correct answer for a premature wrong one.

With 4 passes left, I need to settle two things before the final confirmation on fresh problems: which edit to submit, and what still fails. The most useful open question for the choice is whether E1's head removal still helps now that the "wait" neuron is gone, since E1 carries a number-copying cost.

First I'll look at E3's remaining 63% cut-offs: loop presence and onset, period, the share of "Wait" lines, and how the answers end. That tells me whether verbatim loops (E1's target) are still a major part of what remains.

```python
# What still fails under E3? loops, onset, period, Wait-line share, endings of cut answers; the no-answer-finished case.
cuts3 = [i for i in common if e3_by[i]['cut']]
prof3 = [loop_profile(e3_by[i]['text']) for i in cuts3]
pp3 = [p for p in prof3 if p is not None]
ons3 = [tok_onset(e3_by[i]['text']) for i in cuts3]; f3 = [o for o in ons3 if o is not None]
print(f'E3 cut {len(cuts3)}: verbatim loops {len(pp3)} onset median {np.median(f3):.0f} IQR {np.percentile(f3,25):.0f}-{np.percentile(f3,75):.0f} | period median {np.median([p[0] for p in pp3]):.0f} | Wait-line share {np.mean([p[1] for p in pp3]):.2f}')
ws = sum(1 for i in cuts3 if re.search(r'(\n\s*){40,}$', e3_by[i]['text'][-400:]) is not None)
print('cut answers ending in whitespace runs:', ws)
# top loop line types (first 3 words) in E3 cut answers
from collections import Counter
cnt = Counter()
for i in cuts3:
    t = e3_by[i]['text']; on = loop_onset_char(t)
    if on is None: continue
    seg = [l.strip() for l in t[on:].split('\n') if l.strip()][-40:]
    if seg: cnt[' '.join(Counter(seg).most_common(1)[0][0].split()[:2])] += 1
print('most common loop-line openings:', [(k.encode('ascii','replace').decode()[:25], v) for k, v in cnt.most_common(10)])
print('answer found before loop (E3 cut):', end=' ')
fb = 0
for i in cuts3:
    t = e3_by[i]['text']; on = loop_onset_char(t); ans = str(next(r for r in g8 if r['id'] == i)['answer']).strip()
    m = re.search(r'(?:answer is|=|is)\s*\$?\\?(?:boxed\{)?' + re.escape(ans) + r'(?![0-9])', t)
    fb += (m is not None and on is not None and m.start() < on)
print(fb, '/', len(cuts3))
naf = [i for i in common if not e3_by[i]['cut'] and e3_by[i]['pred'] is None]
print('no-answer finished:', naf, [e3_by[i]['text'][-200:].replace('\n', ' / ').encode('ascii', 'replace').decode() for i in naf])
```