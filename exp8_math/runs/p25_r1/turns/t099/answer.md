The weight-only OV analysis separates the heads cleanly into two groups. Strong positive copiers sit in early and middle layers (L7H2 +8.9, L11H7 +8.7, L11H4 +8.2, L3H0 +6.4, L7H1 +5.7, L15H1 +5.6, L11H3 +5.0; L11H1, an induction head from pass 2, +3.9). There is also a distinct set of copy-suppression heads, whose output pushes down the very token they attend to: L7H0 −10.3, L11H6 −10.0, L7H3 −6.8, L23H2 −6.5, L11H0 −6.0, L3H4 −4.8, L15H3 −4.0, L3H1 −3.3. The deep induction heads (L15H7, L23H5) show no raw-embedding copying, which is expected because the representations they read are far from the input embedding. A copy-suppression head only acts against loops if, inside a loop, it attends to the token that would be copied next (offset i−p+1, as induction heads do) or to the repeated token itself. I will measure these heads' attention in the two loop texts from pass 2. For every head, I compute the attention mass on i−kp+1 (the next-token copy target) and on i−kp (the same token one cycle back) across the loop cycles. A suppression head that tracks the copy target is the network's own anti-repetition brake, a natural candidate to strengthen as n573 was. If these heads look elsewhere (previous token, BOS), they are irrelevant to loops.

```python
# Do the copy-suppression heads (negative OV) attend to the loop's copy target (i-kp+1) or the repeated token (i-kp)? pass-2 attention, original model.
ids2 = np.load('/work/fwd_2/input_ids.npy'); m2 = json.load(open('/work/fwd_2/meta.json')); lens2 = json.loads(m2['lengths']) if isinstance(m2['lengths'], str) else m2['lengths']
A = {L: np.load(f'/work/fwd_2/attn_{L}.npy', mmap_mode='r') for L in [3,7,11,15,19,23]}
print({L: A[L].shape for L in A})
ov = {L: {h: z for h, rk, z, n in ov_scores(L)} for L in [3,7,11,15,19,23]}
def period_and_onset(a):
    n = len(a)
    for p in range(2, 400):
        if np.array_equal(a[n-3*p:], a[n-4*p:n-p]):
            mism = np.where(a[p:] != a[:-p])[0] + p
            return p, int(mism.max())+1
for ti in range(2):
    a = ids2[ti, :lens2[ti]]; p, on = period_and_onset(a); print(f'text{ti}: len {lens2[ti]}, period {p}, onset {on}')
    pos = np.arange(on + p, lens2[ti] - 1)                 # positions inside the loop with at least 1 full previous cycle
    res = []
    for L in A:
        att = np.asarray(A[L][ti])                           # (8,T,T)
        for h in range(8):
            nxt = np.mean([sum(att[h, i, i-k*p+1] for k in range(1,4) if i-k*p+1 >= 0) for i in pos])
            same = np.mean([sum(att[h, i, i-k*p] for k in range(1,4) if i-k*p >= 0) for i in pos])
            prev = np.mean([att[h, i, i-1] for i in pos]); bos = np.mean([att[h, i, 0] for i in pos])
            res.append((L, h, ov[L][h], nxt, same, prev, bos))
    print('  heads with OV z < -3 (copy-suppression):')
    for L,h,z,nx,sm,pv,b in res:
        if z < -3: print(f'   L{L}H{h} OVz {z:+.1f}: mass on i-kp+1 {nx:.2f}, on i-kp {sm:.2f}, prev {pv:.2f}, bos {b:.2f}')
    print('  top i-kp+1 heads:', [(f'L{L}H{h}', round(z,1), round(nx,2)) for L,h,z,nx,sm,pv,b in sorted(res, key=lambda x: -x[3])[:8]])
```