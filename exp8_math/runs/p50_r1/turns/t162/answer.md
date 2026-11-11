The edit is exactly what Pass 13 needs:
- L22 down_proj: only column 573 is changed, at exactly ×2.
- L23 o_proj: the blocks of heads 0, 4, 5 and 7 are zeroed and all other heads are untouched.
- The D1[:256] ids and the gen_11 brake-×2 baseline (256 items × 2 samples) are loaded for pairing.

The forward tool needs the problem list written inline, so I will print the 256 ids as compact JSON to paste into the generation request. The test in Pass 13 is whether silencing the L23 copy heads adds accuracy beyond the brake alone. Teacher-forced passes showed this edit sharply lowers the chance that a loop cycle is copied verbatim. Pairing by seed with gen_11 will tell whether that turns into more finished, correct answers or only into broken formatting.

```python
# Print D1[:256] ids compactly so they can be pasted into the Pass-13 generation request
print(json.dumps(ids,separators=(',',':')))
```