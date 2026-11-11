On the same seeds, 11 samples are right under both models, 20 flipped from wrong to right, and 45 flipped from right to wrong. That is a clear net loss of about 4.9 points, so full ablation of these four heads is rejected.

I want to understand why, because it decides the next direction. I'll apply the existing loop detector (`loop_stats2`) to the edited cut-off samples and compare onset, period and share ending in verbatim repetition with the original. I'll also measure how often edited samples reach the correct number before being cut off.

- If edited samples still loop, but later, with shorter periods or with smaller units, copying is distributed across many heads and the four heads were mainly carrying useful copying.
- If they ramble without verbatim repetition, non-commitment is the real bottleneck.

```python
# Inspect loop detector, then compare loop prevalence/onset/period between original (gen_2) and edited (gen_7).
import inspect
print(inspect.getsource(loop_stats2)[:1500])
```