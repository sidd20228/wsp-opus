The weight scan found no glaring anomaly. Layernorm gains, DeltaNet decay ranges, MLP output norms and attention gains all vary smoothly. So the looping is probably the model's intrinsic behaviour under these sampling settings, not one broken parameter, and I need to localize it from activations. Pass 1 already gives free correlational data: per-answer mean MLP activations over the first 8,192 tokens, where cut answers are mostly in their loop. I'll compare cut and finished answers neuron by neuron, standardized within difficulty strata (level 1–3 and GSM8K, where both outcomes are common) to reduce the confound that harder problems loop more. If a small set of neurons in particular layers shows very large, consistent shifts, they become candidate "repetition neurons" to test causally. Small, diffuse differences would point to the copying being done by attention or DeltaNet rather than by MLP neurons.

```python
# Correlational screen (pass-1 acts): which MLP neurons / layers differ most between looping (cut) and finished answers?
A=json.load(open('/work/gen_1_acts/index.json'))
print(type(A), (list(A.keys()) if isinstance(A,dict) else A[:2]))
```