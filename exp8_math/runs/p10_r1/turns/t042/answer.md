Both texts are pure ASCII, about 4.2k and 5.2k characters when JSON-escaped, so they can be pasted exactly. Each has a natural verbatim loop near the end of its 1,900 tokens:
- **Text 1:** a period-1 loop "- $3^2+3^2=18$.", starting at token 1,435, about 30 iterations.
- **Text 2:** a "Wait, $111$ minutes is ..." loop, starting at token 1,693, about 8 iterations.

The 1,400–1,700 tokens before each loop give a non-loop baseline in the same context. That allows a within-text contrast (loop versus pre-loop positions), which avoids the easy-versus-hard confound I found in pass 1's per-answer averages. I'll print the exact escaped strings now so I can paste them into the teacher-forced request. If the copy mechanism lives in the full-attention layers, pass 2 should show some heads putting most of their attention on the previous loop iteration only inside the loop.

```python
# Print exact JSON-escaped texts (single lines, so the harness won't collapse repeats) for pasting into TF pass 2.
for i, t in enumerate(items):
    print(f'T{i+1}:' + json.dumps(t, ensure_ascii=True))
```