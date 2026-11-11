The divergence at position 0 is suspicious (both texts start with the same prompt), so it is probably a storage detail rather than my paste. Possibly pass 2's first row is padded differently, or the server inserted something. I'll print the first few token ids of each row and the padding at the ends to understand the layout before comparing.

```python
# Setup: inspect padding/layout of input_ids in both passes.
print('p2 t0 head', ids2[0,:6].tolist(), 'tail', ids2[0,830:840].tolist())
print('p4 t0 head', ids4[0,:6].tolist(), 'tail', ids4[0,880:890].tolist())
print('p2 t1 head', ids2[1,:6].tolist())
print(json.load(open(D2+'meta.json')))
```