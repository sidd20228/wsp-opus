Three findings shape the next step:

1. **E1 changes the shape of loops, not their existence.** Short loops of one or two lines fall from 43% to 25% of loops, and the median period grows from 3 to 4 lines. But the remaining loops are even more dominated by "Wait" lines (82% of loop lines start with "Wait", up from 64%).
2. **The doubt spiral is untouched.** In cut-off answers the "Wait" rate still climbs with length, from about 4 to 42 per 1,000 tokens. So the growing "Wait" rate is not driven by the three heads I removed.
3. **The non-loop cut-offs are near-loops.** The 57 cut-off answers without a detected verbatim loop are degenerate in a similar way: endless blank lines, or alternating cycles such as "Wait, looking at the user message again. / Okay, I'll write the solution." My line detector simply doesn't catch them.

So the residual failure is still repetition, now carried by line-start "Wait" copying spread over other heads. In the E1 teacher-forced pass, text 2's " Wait" decision margin still grew from 0.5 to 2.2 over the loop.

I'll find which of the remaining layer-19 and layer-23 heads attend to earlier " Wait" tokens at those line-start decision points, and whether that attention grows with each repetition, in both the original and E1 attention patterns. A small set of backup heads whose attention on earlier " Wait"s grows across iterations would be the targets for a second edit. If none stands out, the remaining drive sits in DeltaNet layers or MLPs.

```python
# Which remaining heads (L19/L23) copy line-start ' Wait' at T2 decision points? attention on earlier ' Wait' tokens, orig vs E1, first vs last iterations.
i = 1
dp = dp_positions(i)
wait_pos = np.where(np.isin(ids[i], [13428, 13784]))[0]
print('T2 decision pts', len(dp), '| earlier Wait tokens before first dp:', (wait_pos < dp[0]).sum())
for L in [19, 23]:
    A0 = np.load(D + f'attn_{L}.npy', mmap_mode='r'); A1 = np.load(D5 + f'attn_{L}.npy', mmap_mode='r')
    line = []
    for h in range(8):
        a0 = np.asarray(A0[i, h]); a1 = np.asarray(A1[i, h])
        m0 = [a0[t, wait_pos[wait_pos < t]].sum() for t in dp]; m1 = [a1[t, wait_pos[wait_pos < t]].sum() for t in dp]
        line.append(f'{L}.{h}: o {m0[0]:.2f}->{m0[-1]:.2f} | e {m1[0]:.2f}->{m1[-1]:.2f}')
    print('\n'.join(line))
# also original layers 7, 11, 15 (orig only)
for L in [7, 11, 15]:
    A0 = np.load(D + f'attn_{L}.npy', mmap_mode='r')
    out = []
    for h in range(8):
        a0 = np.asarray(A0[i, h]); m0 = [a0[t, wait_pos[wait_pos < t]].sum() for t in dp]
        out.append(f'{L}.{h}: {m0[0]:.2f}->{m0[-1]:.2f}')
    print(' | '.join(out))
```