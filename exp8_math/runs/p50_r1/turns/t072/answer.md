The ablation is causal but small. With layer-16 neuron 998 zeroed:

| Text | P(Wait) before loop, orig → ablated | P(Wait) in loop, orig → ablated |
|---|---|---|
| A | 0.35 → 0.22 (−0.64 nats) | 0.97 → 0.96 |
| B | 0.62 → 0.48 (−0.35 nats) | 0.99 → 0.98 |

All non-Wait tokens are unchanged (mean Δlog p ≈ 0.000). So the neuron is a clean, specific "Wait" writer. However, roughly 80–90% of its +3-nat direct effect is compensated downstream: the layer-22 MLP, which pushes against "Wait" by about 4 nats, seems to act as negative feedback. Inside loops, copying sustains "Wait" regardless. On its own, this edit is probably too weak to change outcomes. Making it stronger (for example, flipping the neuron's sign) would be a blunt anti-Wait knob rather than a repair.

The network may also contain an actual malfunction rather than just an over-strong habit. The hazard profile is suggestive: almost no looping before about 2k tokens, then a steady 11–17% per 1k tokens. Two candidates fit a mechanism that breaks with context length. One is the Gated DeltaNet layers' decay and step-size parameters (A_log, dt_bias), which control how fast their memory fades. The other is the full-attention QK-norm scales, which control how sharp copying attention can get. Earlier I only screened projection norms. So I'll list each layer type's tensors and screen these small parameters across layers for outliers, such as one layer whose decay or attention sharpness is far out of line with its neighbours.

```python
# Look for an anomaly in long-context parameters: list tensor names of one DeltaNet and one full-attn layer; per-layer stats of A_log, dt_bias, q/k norms.
print([n for n in NAMES if '.layers.0.' in n and 'language_model' in n])
print([n for n in NAMES if '.layers.3.' in n and 'language_model' in n])
for l in range(24):
    pre_ = f'model.language_model.layers.{l}.'
    if pre_ + 'linear_attn.A_log' in NAMES:
        A_ = load(pre_ + 'linear_attn.A_log').numpy(); dt_ = load(pre_ + 'linear_attn.dt_bias').numpy()
        print('L%2d DeltaNet A_log mean %.2f [%.2f,%.2f] | dt_bias mean %.2f [%.2f,%.2f]' % (l, A_.mean(), A_.min(), A_.max(), dt_.mean(), dt_.min(), dt_.max()))
    else:
        qn_ = load(pre_ + 'self_attn.q_norm.weight').numpy(); kn_ = load(pre_ + 'self_attn.k_norm.weight').numpy()
        print('L%2d FullAttn q_norm mean %.3f [%.2f,%.2f] | k_norm mean %.3f [%.2f,%.2f]' % (l, qn_.mean(), qn_.min(), qn_.max(), kn_.mean(), kn_.min(), kn_.max()))
```