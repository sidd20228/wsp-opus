Brake ×3 gives 43.6% accuracy (×2: 43.0%) and 53.7% cut (×2: 55.9%), with only 20 wrong finished answers. On the headline numbers the dose-response has flattened between ×2 and ×3. That fits the earlier finding that only about 10% of the surviving loop units contain doubt words: the brake has mostly done its job, and the remaining loops have a different driver.

Before concluding, this run computes the paired difference against gen_11 (×2) and gen_2 (original) and the same hazard table, and counts 'wait'/'Wait' per 1k thinking tokens in each run. That shows whether ×3 still suppresses doubts more while the loop hazard stays flat. I'll keep ×2 or ×3 depending on the paired result. Then I'll turn to what starts the restatement loops.

```python
# Check: x3 vs x2 paired; hazards; doubt-word rate. Hypothesis: brake saturates; residual loops are not doubt-driven.
g14 = readgen('/work/gen_14.jsonl'); F14 = flat(g14)
IDS14 = {k: tok.encode(F14[k]['text']).ids for k in keys}; LI14 = {k: loop_info(IDS14[k]) for k in keys}
for nm,Fa,Fb in [('x3-x2',F14,F11),('x3-orig',F14,F2)]:
    dd=np.array([Fa[k]['correct']-Fb[k]['correct'] for k in keys],float)
    print(f"{nm}: {dd.mean()*100:+.1f} (SE {dd.std(ddof=1)/np.sqrt(len(dd))*100:.1f}) w->r {(dd>0).sum()} r->w {(dd<0).sum()}")
fin=[k for k in keys if not F14[k]['cut']]
print("x3 fin", len(fin), "fin-acc %.1f"%(np.mean([F14[k]['correct'] for k in fin])*100), "fin ntok med", np.median([F14[k]['tokens'] for k in fin]))
for a,b,r,f,l,hf,hl in hazards(events(F14,LI14),edges): print(f"  {a:>2}-{b:<2} {r:4d} {f:4d} {l:4d} | {hf:.3f} {hl:.3f}")
def wrate(F,IDS,LI):
    # 'wait' occurrences per 1k tokens in the pre-loop part of the thinking
    r=[]
    for k in keys:
        t=F[k]['text']; end = int(LI[k]['onset']) if LI[k]['looped'] else len(IDS[k])
        pre = tok.decode(IDS[k][:end]); r.append(len(re.findall(r'\b[Ww]ait\b',pre))/max(end,1)*1000)
    return np.mean(r)
print("wait/1k pre-loop: orig %.2f x2 %.2f x3 %.2f"%(wrate(F2,IDS2,LI2),wrate(F11,IDS11,LI11),wrate(F14,IDS14,LI14)))
```