These attribution numbers can't be taken at face value. Layer 23's "mixer" term is hugely negative (−1.9 inside the loop, −7.1 before it), and all other contributions are tiny. The likely cause: like HuggingFace's convention, the last recorded state (`hidden_24`) is taken after the final RMSNorm. Then "hidden_24 − hidden_23" mixes layer 23's update with the normalisation, and my scale factor (RMS of `hidden_24`) is wrong for every layer.

Comparisons among layers 0–22 still hold up to a common scale. On that basis:

- The layer-19 mixer (+0.25 in the loop vs +0.04 before it) and the layer-15 mixer (+0.09 vs +0.03), where the induction heads sit, push toward "Wait" mainly inside the loop.
- The layer-16 MLP pushes toward "Wait" both inside and before the loop (+0.18).
- The layer-22 and layer-23 MLPs push against it.

To fix the scale, I'll check which convention reproduces the recorded log-probabilities. I'll compare softmax(E·h24) and softmax(E·norm(h24)) at a few positions against `tlp`. I'll also compare the RMS of `hidden_23` and `hidden_24`. If `hidden_24` is post-norm, I'll redo the attribution with the correct pre-norm scale.

```python
# Check convention: is hidden_24 post-final-norm? Compare log_softmax(E h24) vs log_softmax(E norm(h24)) with recorded token logprobs.
Et_ = Ew  # (V,1024) torch
chk = [(0, 300), (0, 1200), (1, 500), (1, 1500)]
for i, t in chk:
    h24 = torch.tensor(Hs[24][i, t]); h23 = torch.tensor(Hs[23][i, t])
    lgA = torch.log_softmax(Et_ @ h24, -1); lgB = torch.log_softmax(Et_ @ (torch.tensor(gfin, dtype=torch.float32) * h24 / h24.pow(2).mean().add(1e-6).sqrt()), -1)
    nxt = int(ids5[i, t + 1])
    print('pos', i, t, '| recorded %.3f | E.h24 %.3f | E.norm(h24) %.3f | rms h23 %.2f h24 %.2f' % (tlp[i, t], lgA[nxt], lgB[nxt], h23.pow(2).mean().sqrt(), h24.pow(2).mean().sqrt()))
```