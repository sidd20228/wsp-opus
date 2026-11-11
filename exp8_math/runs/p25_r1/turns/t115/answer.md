The gate edit is built. In-sample, ordinary positions score within −0.18…+0.17 (1st–99th percentile) and junctions about 0.75–1.0 at all four layers. Only the four q_proj tensors are edited for now, with no brake and no E1, so the teacher-forced check compares directly with the original model's saved log-probabilities. The five check texts total about 10k characters of JSON, more than one output can show, so I print them ASCII-escaped in two parts and paste them exactly into the request. Texts 0–1 are the two loop texts.

```python
# Print TF-check texts 0-1 (the two loop texts) JSON-escaped (ASCII-safe) for pasting into the forward block.
print(json.dumps(tfc[:2], ensure_ascii=True))
```