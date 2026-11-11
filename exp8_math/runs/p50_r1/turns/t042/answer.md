Run 34 recovered the context. `fwd_6` is the teacher-forced pass with the four induction heads (L11H1, L15H7, L19H5, L23H5) zeroed. It returned only token log-probabilities and top-k, and the kernel holds them as `tlp6`, with the original's values in `tlp`. The run-33 output therefore came from `dirs`, a per-layer "loop minus pre-loop" mean-difference direction computed on the original hidden states. It was not an edit effect.

The main question is still unanswered: does the ablation weaken loop continuation specifically? I'll compare mean token log-probabilities between original and edited, inside the loop region versus before the loop, for both texts. I'll also look at the summary tables already in memory (`S`, `NN`, `stats`) so I don't redo earlier work.

The ablation is causally responsible for loop copying if the in-loop log-probability drops a lot, toward pre-loop levels, while pre-loop tokens barely change. If both regions drop similarly, the heads are a general copying resource and zeroing them would just damage the model.

```python
# Hypothesis: zeroing the 4 induction heads lowers logprob of loop continuation (in-loop tokens) much more than pre-loop tokens.
import numpy as np, pandas as pd
print('loop_start_tok', loop_start_tok, 'offA', offA, 'offB', offB, 'lens5', lens5)
print('tlp', tlp.shape, 'tlp6', tlp6.shape)
for i,nm_ in enumerate(['A','B']):
    L_ = lens5[i]; ls = loop_start_tok[i] if isinstance(loop_start_tok, dict) and i in loop_start_tok else list(loop_start_tok.values())[i]
    o5 = tlp[i,:L_-1]; o6 = tlp6[i,:L_-1]
    pre = slice(200, ls-1); lp_ = slice(ls+50, L_-1)
    print(nm_, 'loop_start', ls,
          '| pre-loop mean lp orig %.3f edit %.3f' % (o5[pre].mean(), o6[pre].mean()),
          '| in-loop mean lp orig %.3f edit %.3f' % (o5[lp_].mean(), o6[lp_].mean()),
          '| in-loop P(next) orig %.3f edit %.3f' % (np.exp(o5[lp_]).mean(), np.exp(o6[lp_]).mean()))
    # tokens where edited assigns <0.5 prob in loop region (potential escape points)
    print('   frac in-loop tokens with p<0.5: orig %.3f edit %.3f ; p<0.9: orig %.3f edit %.3f' % ((np.exp(o5[lp_])<.5).mean(), (np.exp(o6[lp_])<.5).mean(), (np.exp(o5[lp_])<.9).mean(), (np.exp(o6[lp_])<.9).mean()))
for nm_ in ['S','NN','R2','Rm']:
    v = globals().get(nm_)
    if v is not None: print(nm_, v.shape); print(v.head(12).to_string()[:1500])
print(type(stats), len(stats), str(stats[:3])[:600])
print('dirs keys', list(dirs.keys())[:30] if isinstance(dirs, dict) else type(dirs))
```