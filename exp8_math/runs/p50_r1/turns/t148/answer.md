The six probe texts look right; the GSM8K solutions in this file simply have no "#### N" line, so nothing had to be rewritten. Next I'll print the JSON of the two truncated loop texts (identical to fwd_10's inputs). The first screening pass will run the current edit (brake ×2 only) on all eight texts, returning token log-probs and top-20. That gives the paired baseline for in-loop per-cycle copy probability and for the clean-text log-likelihood, which each L23 copy-head variant will be compared against.

```python
# Print JSON of the two loop texts (same token ids as fwd_10) for the TF baseline pass
print(json.dumps(TT))
```