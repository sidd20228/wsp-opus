The calibration shows the cost of E1 is real. When only the brake dose changed (x3→x5), 4 of 215 correct answers became wrong. With E1, 19 of 228 did, four to five times the churn. Those are ordinary reasoning slips (180→160, 36→20, 1331→216) of moderate length, so the induction heads partly carry legitimate copying of numbers and expressions. The rescued loops dominate (79 cut→correct), hence the net +6.2. The next question is whether removing more induction capacity keeps paying off, and which heads give the best ratio of loop-breaking to collateral. In the pass-2 loop texts, I will rank every full-attention head by its attention to the copy target inside loops (i−kp+1). I will compare that with its classic induction score in the healthy text before the loop (attention to the token after an earlier occurrence of the current token). A head that copies heavily in loops but little in normal reasoning is a good next target. One that does both will mostly add collateral.

```python
# Rank FA heads: in-loop copy-target attention vs induction use in healthy pre-onset text (fwd_2 text1 has ~480 healthy tokens). Loop-specialized heads = better targets.
a1 = ids2[1, :lens2[1]]; p1, on1 = period_and_onset(a1)
prompt_len = int(np.where(a1 == tok.encode('<think>').ids[0])[0][0]) if (a1 == tok.encode('<think>').ids[0]).any() else 60
print('text1 prompt/think at', prompt_len, 'onset', on1, 'period', p1)
def induction_score(att_h, a, lo, hi):
    sc = []
    for i in range(lo, hi):
        prev = np.where(a[:i-1] == a[i])[0]          # earlier occurrences of current token
        prev = prev[prev + 1 < i]
        if len(prev) == 0: continue
        sc.append(att_h[i, prev + 1].sum())
    return np.mean(sc) if sc else np.nan, len(sc)
rows_h = []
for L in A:
    att1 = np.asarray(A[L][1]); att0 = np.asarray(A[L][0]); a0 = ids2[0, :lens2[0]]; p0, on0 = period_and_onset(a0)
    for h in range(8):
        loop1 = np.mean([sum(att1[h, i, i-k*p1+1] for k in range(1,4) if i-k*p1+1 >= 0) for i in range(on1+p1, lens2[1]-1)])
        loop0 = np.mean([sum(att0[h, i, i-k*p0+1] for k in range(1,4) if i-k*p0+1 >= 0) for i in range(on0+p0, lens2[0]-1)])
        ind_pre, npos = induction_score(att1[h], a1, prompt_len + 20, on1)
        rows_h.append((L, h, loop0, loop1, ind_pre, npos, ov[L][h]))
rows_h.sort(key=lambda x: -(x[2]+x[3]))
print('head   loop0  loop1  healthy-induction  OVz')
for L,h,l0,l1,ip,npn,z in rows_h[:16]: print(f'L{L}H{h}  {l0:.2f}   {l1:.2f}   {ip:.2f} (n={npn})   {z:+.1f}')
```