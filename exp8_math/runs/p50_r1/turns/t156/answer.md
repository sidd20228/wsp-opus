Variant A shows a dose-dependent effect in the predicted direction, at a small cost. The late-cycle copy probabilities drop for two of the three loops:
- GSM8K: copy 3 0.55 → 0.36, copy 4 0.79 → 0.58, copy 5 0.93 → 0.80
- Geometry: copy 2 0.55 → 0.16, copy 3 0.88 → 0.60

The algebra loop barely moves (copy 3 0.55 → 0.49). On the reference solutions the clean-text cost is only −0.017 nats per token, and pre-loop text changes by ±0.03. So L23's copy heads do carry part of the cycle-to-cycle self-reinforcement, but halving them only delays lock-in by about one cycle; by copies 5–7 the probabilities are back at 0.8–0.98.

One caveat: whole-cycle copy probability is only a proxy. A single deviating token can rejoin the loop, so escape needs a real divergence.

To complete the dose-response, I'll test the stronger setting: the same four heads (H0, H4, H5, H7) fully removed (×0), everything else unchanged. If lock-in in copies 3–6 then drops much further (for example geometry copy 3 well below 0.6, GSM8K copy 5 below 0.6) while clean-text cost stays under about −0.05 nats per token, the strong version is worth a generation test. If the curve flattens or the cost jumps, the moderate version or a different lever is preferable.

```python
# Variant B: brake x2 + L23 copy heads H0,H4,H5,H7 fully removed (x0) -> dose-response of lock-in softening vs clean-text cost
Wv = set_l23_heads({0:0.0,4:0.0,5:0.0,7:0.0})
chk = load('model.language_model.layers.23.self_attn.o_proj.weight')
print(edited(), [round(float(chk[:,h*256:(h+1)*256].norm()/Wo23[:,h*256:(h+1)*256].norm()),2) for h in range(8)])
```