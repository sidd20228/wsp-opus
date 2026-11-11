Variant B (four L23 copy heads fully removed) has finished. Its whole-text mean log-probs move a little more than A's, both up and down. This run applies the same paired comparison against the brake-only baseline: per-cycle copy probability, per-cycle change in log-prob, pre-loop change, and the clean-text cost on reference solutions. With A in hand, that gives a three-point dose-response (1.0, 0.5, 0.0) for the four heads.

```python
# Dose-response of L23 copy heads (H0,4,5,7): x1 (baseline) -> x0.5 (A) -> x0 (B): lock-in per cycle and clean-text cost
T17=np.load('/work/fwd_17/token_logprobs.npy'); print("ids identical:", bool((np.load('/work/fwd_17/input_ids.npy')==I15).all()))
compare_variant(T17, "B: L23 H0,4,5,7 x0")
```