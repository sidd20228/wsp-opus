Two useful facts come out of the matched comparisons. First, brake strength does not hurt accuracy on answers that finish either way: ×3 vs ×5 gives 92.6 vs 93.2 (−0.6 ± 1.7), so the brake is not removing useful self-correction. Second, the gate does cost accuracy on answers that would have finished without it: x5+gate vs x5 is −5.5 ± 2.2 on 255 matched finished samples. The gate's net gain comes from rescuing loops, at a real price on clean answers. Some of that price is the \boxed format failure. I'll decompose the matched losses into format failures (noans) versus genuinely wrong answers, and read a few of the gate-induced right→wrong cases. If they are mostly format, a format repair is worth about 3 points. If they are reasoning errors, I'll look for what the gate breaks inside lines of correct reasoning, such as restating earlier numbers at line starts.

```python
# Decompose the gate's cost on matched finished samples (x5 vs x5+gate): format (noans) vs wrong; peek at ok->wrong cases.
ks=[k for k in T12 if T8[k]['st']!='cut' and T12[k]['st']!='cut']
trm=collections.Counter((T8[k]['st'],T12[k]['st']) for k in ks); print('x5 -> x5+gate (both finished):', sorted(trm.items(), key=lambda x:-x[1]))
okwr=[k for k in ks if T8[k]['st']=='ok' and T12[k]['st']=='wrong']
for k in okwr[:6]:
    s12=byid[k[0]]['samples'][k[1]]; s8=[r for r in g8 if r['id']==k[0]][0]['samples'][k[1]]
    print(f"\n### {k[0]} ref={ref[k[0]]} x5 pred={s8['pred']} tok={s8['tokens']} | gate pred={s12['pred']} tok={s12['tokens']}")
    print('   gate end:', asc(s12['text'].split('</think>')[0][-220:]).replace('\n',' | '))
```