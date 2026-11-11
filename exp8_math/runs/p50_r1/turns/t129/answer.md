Brake ×3 is saved (column 573 ×3.0, no other columns touched). The forward block needs the 256 D1 item ids written out, so I'll print the full JSON list to paste in. If it goes over the 10,000-character output limit I'll print it in two halves.

```python
# Print the D1[:256] id list (for pasting into the forward block)
sjs = json.dumps(D1_256, separators=(',',':')); print(len(sjs)); print(sjs if len(sjs)<9800 else sjs[:4900])
```