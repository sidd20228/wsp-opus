Variant A changed the mean log-probs very little. The reference texts even improved slightly (for example −1.609 → −1.562), so the clean-text cost is small or negative. The key question is whether the late-cycle lock-in softened. This run recomputes the per-cycle copy statistics for variant A and sets them next to the baseline, and gives the paired change in solution-token log-prob on the six reference texts. It also gives the paired change in log-prob within each cycle, which shows directly how much the copy confidence dropped as repeats accumulate.

```python
# Compare variant A (L23 H0,4,5,7 x0.5) vs baseline x2 on lock-in per cycle and clean-text cost
T16=np.load('/work/fwd_16/token_logprobs.npy'); I16=np.load('/work/fwd_16/input_ids.npy')
print("ids identical to baseline:", bool((I16==I15).all()))
def compare_variant(TV, name):
    print(f"== {name}")
    for i,m in enumerate(LTmeta):
        b = cycle_stats(T15,i,m['on_in_text'],m['period']); v=cycle_stats(TV,i,m['on_in_text'],m['period'])
        dl = [float((TV[i,m['on_in_text']+c*m['period']-1:m['on_in_text']+(c+1)*m['period']-1]-T15[i,m['on_in_text']+c*m['period']-1:m['on_in_text']+(c+1)*m['period']-1]).mean()) for c in range(8)]
        print("  ", m['k'][0][11:30], " ".join(f"c{c}:{b[c][1]:.2f}->{v[c][1]:.2f}" for c in range(1,8)))
        print("      dlogp/token per cycle:", " ".join(f"{x:+.3f}" for x in dl), "| pre-loop", f"{float((TV[i,100:m['on_in_text']-1]-T15[i,100:m['on_in_text']-1]).mean()):+.3f}")
    d=[float((TV[3+j,prompt_len[j]-1:L15[3+j]-1]-T15[3+j,prompt_len[j]-1:L15[3+j]-1]).mean()) for j in range(6)]
    print("   ref-solution dlogp:", np.round(d,3), "mean %+.3f"%np.mean(d))
compare_variant(T16, "A: L23 H0,4,5,7 x0.5")
```