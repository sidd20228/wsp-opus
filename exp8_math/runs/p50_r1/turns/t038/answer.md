The ablated pass ran, but text B came back 43 tokens shorter (1,785 vs 1,828), so I evidently dropped a few repeated lines when re-pasting it. Text A is identical (1,873 tokens). The whole-text mean log-prob fell from −0.31 to −0.39 (A) and from −0.18 to −0.37 (B). I'll compare token by token. For A, that's a direct position-wise comparison. For B, I'll use the identical prefix, which covers the pre-loop part and the start of the loop. I'll split the effect into pre-loop tokens, loop tokens, and loop line-start tokens. If loop tokens lose much more log-prob than pre-loop tokens, these heads are causal drivers of the copy.

```python
# Hypothesis: ablating 4 induction heads lowers loop-token logprob much more than pre-loop-token logprob.
ids6 = np.load('/work/fwd_6/input_ids.npy'); tlp6 = np.load('/work/fwd_6/token_logprobs.npy')
def compare(b, tlp_new, ids_new, label):
    L0 = lens5[b]; a = ids5[b, :L0]; n = ids_new[b, :L0]
    m = min(L0, len(n)); same = np.where(a[:m] != n[:m])[0]; common = same[0] if len(same) else m
    on = loop_start_tok[b]
    o = tlp[b, :common-1]; e = tlp_new[b, :common-1]   # logprob of token t+1 given prefix up to t
    pos = np.arange(1, common)                          # index of predicted token
    pre = (pos > 150) & (pos < on); lp = pos >= on + 64
    nl = np.array([('\n' in tok.decode([int(a[p-1])])) for p in pos])   # predicted token follows a newline (line start)
    print(f'{label} text{b}: common={common} onset={on}')
    for nm, msk in [('pre-loop', pre), ('loop', lp), ('loop line-start', lp & nl), ('pre line-start', pre & nl)]:
        if msk.sum() == 0: continue
        print(f'   {nm:16s} n={msk.sum():4d}  orig={o[msk].mean():7.3f}  edit={e[msk].mean():7.3f}  diff={(e[msk]-o[msk]).mean():7.3f}  p_orig={np.exp(o[msk]).mean():.3f} p_edit={np.exp(e[msk]).mean():.3f}')
for b in range(2): compare(b, tlp6, ids6, 'ablate IND4')
```