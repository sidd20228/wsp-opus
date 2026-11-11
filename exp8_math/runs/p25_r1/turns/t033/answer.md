Zeroing the three induction heads weakens the loop but does not break it. In the clean loop (text 0), the per-token loss on loop tokens rises about 20-fold (mean logprob −0.001 → −0.024), yet 94% of loop tokens still have p > 0.9. In text 1 the share of near-certain tokens in late cycles drops from ~0.99 to ~0.85. So these heads are causally involved in the copying, but other components (second-tier heads and probably DeltaNet associative recall) carry much of it redundantly. Full-head removal alone looks insufficient and costly. Next I'll look at the "decision points" within each cycle, where the model could leave the loop: the lowest-probability positions and their top alternatives, original vs. edited. If the weak points are where the model chooses "Wait" over concluding, the loop is a stuck self-verification cycle, and the relevant mechanism may be the restart decision as well as the copying.

```python
# Hypothesis: loop exits are possible only at a few decision points (e.g. before 'Wait'); inspect them and their alternatives.
tk4i=np.load(D4+'topk_ids.npy'); tk4l=np.load(D4+'topk_logprobs.npy')
tk2i=np.load(D2+'topk_ids.npy'); tk2l=np.load(D2+'topk_logprobs.npy')
def tokstr(i): return asc(tok.decode([int(i)]))
for t,n_o in enumerate([837,959]):
    p=per2[t]; s=n_o-2*p  # second-to-last full cycle
    le=lp4[t,s-1:s+p-1]; lo=lp2[t,s-1:s+p-1]
    worst=np.argsort(le)[:4]
    print('text',t)
    for w in sorted(worst):
        pos=s-1+w   # predicting token at pos+1
        ctx=tok.decode(ids2[t,max(0,pos-8):pos+1].tolist())
        nxt=tokstr(ids2[t,pos+1])
        alt_o=[(tokstr(a),round(float(np.exp(b)),3)) for a,b in zip(tk2i[t,pos,:4],tk2l[t,pos,:4])]
        alt_e=[(tokstr(a),round(float(np.exp(b)),3)) for a,b in zip(tk4i[t,pos,:4],tk4l[t,pos,:4])]
        print(f'  ctx="{asc(ctx)[-40:]}" next="{nxt}" | orig {alt_o} | edit {alt_e}')
```