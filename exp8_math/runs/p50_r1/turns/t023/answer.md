Both texts are pure ASCII and short (~4.5k and ~3.8k characters JSON-escaped), and both end deep inside the loop. I'll print them as single-line JSON strings so the display doesn't collapse repeated lines, and copy them exactly into the attention pass.

```python
# Hypothesis: (setup) print JSON-escaped texts for the attention pass.
print(json.dumps(TA))
print(json.dumps(TB))
```