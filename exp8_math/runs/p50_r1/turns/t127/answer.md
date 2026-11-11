loop_info returns onset, period, run length and a looped flag, which is what I need. I'll tokenize every sample of gen_2 (original) and gen_11 (×2), cache the token ids and loop info in kernel variables, and compute per-1k-token hazards in a competing-risk setup. A sample counts as at risk in a bin if it has neither finished nor entered its final loop before the bin starts, and the two events are "finished in this bin" and "loop onset in this bin". Comparing the two curves shows what ×2 changed.

```python
# Hypothesis: brake x2 acts by raising the finishing hazard (not by delaying loop onset). Compute competing hazards per 1k tokens.
import time
t0=time.time()
IDS2 = {k: tok.encode(F2[k]['text']).ids for k in keys}
IDS11 = {k: tok.encode(F11[k]['text']).ids for k in keys}
LI2 = {k: loop_info(IDS2[k]) for k in keys}; LI11 = {k: loop_info(IDS11[k]) for k in keys}
print("time %.0fs"%(time.time()-t0))
def events(F,LI):
    ev=[]
    for k in keys:
        s=F[k]; li=LI[k]
        if not s['cut']: ev.append(('fin', s['tokens']))
        elif li['looped']: ev.append(('loop', int(li['onset'])))
        else: ev.append(('cut', s['tokens']))
    return ev
def hazards(ev, edges):
    out=[]
    for a,b in zip(edges[:-1],edges[1:]):
        risk=sum(1 for e,t in ev if t>=a); f=sum(1 for e,t in ev if e=='fin' and a<=t<b); l=sum(1 for e,t in ev if e=='loop' and a<=t<b)
        out.append((a//1000,b//1000,risk,f,l, f/max(risk,1)/((b-a)/1000), l/max(risk,1)/((b-a)/1000)))
    return out
edges=[0,1000,2000,3000,4000,6000,8000,12000,16000,24000,32769]
for nm,F,LI in [('orig',F2,LI2),('x2',F11,LI11)]:
    print(nm, "bin(k) risk fin loop | hz_fin hz_loop per 1k")
    for a,b,r,f,l,hf,hl in hazards(events(F,LI),edges): print(f"  {a:>2}-{b:<2} {r:4d} {f:4d} {l:4d} | {hf:.3f} {hl:.3f}")
```