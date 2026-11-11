N573 is unique. Its activation at the 98 decision points is −5.46, against −0.92 at other line starts and −0.13 elsewhere (−7.7 SD). Its direct effect there is −6.4 logits on 'wait' versus commit openers and −0.46 on the other re-open openers. No other neuron in layers 16–23 has a context-gated effect above about 0.15 logits on the non-'wait' openers. The one big accelerator, L16 N998 (+3.2 on 'wait'), only matters for 'wait', which is already suppressed.

So there is no second existing brake for the remaining re-open words to strengthen. ×3 already raised N573's push against those openers from about −0.9 to −1.4 logits with no accuracy gain. A neuron-level doubt edit has run its course.

That points back to the other half of the mechanism: once a cycle has repeated, it never escapes. My earlier logit-lens analysis found the copy margin goes from 5.7 in the first repeated cycle to 12.75 in late cycles, mostly added by layer 23 attention. If the lock-in builds up over several cycles, weakening the late amplification could keep loops escapable. If it is already complete in the first one or two cycles, attention edits on late cycles won't help.

This run measures, in texts A/B (fwd_5, original weights), the in-loop probability of the actual next token by cycle index. For each cycle I report the fraction of tokens with p < 0.9 and the chance that a whole cycle is copied without any deviation at temperature 0.6.

```python
# Hypothesis: lock-in builds up over cycles (escapable early, absorbing late) -> per-cycle P(copy) and P(cycle copied w/o deviation)
TL5 = np.load('/work/fwd_5/token_logprobs.npy')
for t in range(2):
    ids_t = [int(x) for x in I5[t,:Lt[t]]]
    li = loop_info(ids_t); on, per = int(li['onset']), int(li['period'])
    print(f"text {t}: onset {on} period {per} len {Lt[t]}")
    lp = TL5[t,:Lt[t]-1]   # logprob of token p+1 given prefix up to p
    out=[]
    for c in range(0, (Lt[t]-1-on)//per):
        a,b = on+c*per, on+(c+1)*per
        l_ = lp[a-1:b-1]; p_ = np.exp(l_)
        # approx P(no deviation) at T=0.6: p_T = p^(1/T)/(p^(1/T)+(1-p)^(1/T)) as 2-way approx
        pT = p_**(1/0.6)/(p_**(1/0.6)+(1-p_)**(1/0.6)+1e-12)
        out.append(f"c{c}:{np.mean(p_<.9):.2f}/{np.exp(np.log(pT+1e-12).sum()):.2f}")
    print("   frac p<.9 / P(cycle copied):", " ".join(out[:14]))
```