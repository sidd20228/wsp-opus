Both texts show the same thing: the cycle-to-cycle self-reinforcement happens almost entirely in the last layer. Layer 23's contribution to the copy margin (h24 minus h23) is about −1.1 to +0.2 on normal pre-loop text. It is +0.2 to +1.0 on the first copy, about +1.6 to +1.9 on the second, +2.5 to +2.8 on the third, and +4.4 to +5.1 from the fourth on. Through layer 23's input the margin is nearly flat across cycles in text B (6.8 → 7.3) and grows only moderately in text A (6.3 → 8.6). Layer 16 adds a large copy signal from the first copy onward (L15 holds the strongest induction head, H7). That is ordinary copying and stays constant, not self-reinforcing.

So the mechanism that makes the second and third copies near-certain sits in L23, which damps confidence on normal text but amplifies it more with each previous repeat. That pattern is what softmax attention produces when the number of matching earlier copies grows.

To find which L23 heads carry this, I'll measure from the stored attention patterns (fwd_5) each head's copy mass per cycle: attention from position i to earlier positions holding the token that comes next at i+1. A head whose copy mass rises steadily from pre-loop through c1 to c4+ is the candidate for a targeted edit. Heads with high but flat copy mass are doing normal copying and should be left alone.

```python
# Hypothesis: 1-2 L23 heads carry the cycle-growing copy mass (self-reinforcement); measure copy mass per head x cycle from stored attention
At23 = np.load('/work/fwd_5/attn_23.npy', mmap_mode='r'); print(At23.shape)
for t in range(2):
    ids_t = np.array([int(x) for x in I5[t,:Lt[t]]]); li = loop_info(list(ids_t)); on, per = int(li['onset']), int(li['period'])
    groups = {'pre':list(range(300,on-1)), 'c1':list(range(on+per-1,on+2*per-1)), 'c2':list(range(on+2*per-1,on+3*per-1)),
              'c3':list(range(on+3*per-1,on+4*per-1)), 'c4+':list(range(on+4*per-1,Lt[t]-1))}
    print(f"text {t}: L23 copy mass (attn on earlier positions holding next token) head x group(pre c1 c2 c3 c4+)")
    for h in range(8):
        row=[]
        for g,ps in groups.items():
            ms=[]
            for i in (ps[::3] if len(ps)>150 else ps):
                nx = ids_t[i+1]; js = np.nonzero(ids_t[:i]==nx)[0]
                ms.append(float(np.asarray(At23[t,h,i,js]).sum()) if len(js) else 0.0)
            row.append(np.mean(ms))
        print(f"  H{h}: " + "  ".join(f"{v:.2f}" for v in row))
```