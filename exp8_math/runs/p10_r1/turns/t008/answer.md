This changes the picture completely. Accuracy is only 12.3%, and 89% of answers (448 of 512) hit the 32,768-token limit. Only one finished answer was wrong, so the 64 answers that finish are about 98% correct. The model can solve these problems but almost never stops thinking, even on grade-school GSM8K questions. That is not normal for this model family; it looks like a specific broken mechanism, such as an inability to emit `</think>` or a degenerate loop. Next I'll read the generation file's structure, the length distribution, and short excerpts of how cut-off answers begin and end. That will show whether they loop, drift into garbage, or solve the problem and then fail to close the thinking.

```python
# Diagnose pass-1 failures: structure of gen_1.jsonl, lengths, whether </think> appears, excerpts
rows = [json.loads(l) for l in open('/work/gen_1.jsonl')]
print(rows[0].keys())
r0 = rows[0]
print({k:(v if not isinstance(v,str) or len(v)<100 else v[:100]) for k,v in r0.items()})
```