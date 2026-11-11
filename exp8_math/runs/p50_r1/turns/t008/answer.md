The tool needs explicit practice ids, not a split name, so the failed request wasn't counted. The full D1 list (~14k characters) exceeds the 10k output limit, so I'll print it in two halves and paste the ids into the forward block.

```python
# Hypothesis: none - print D1 ids (first half) compactly for the forward block.
print(json.dumps(splits['D1'][:256], separators=(',', ':')))
```