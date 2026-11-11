This is a striking result: accuracy is only 11.5%, and 89% of answers (450 of 512) hit the 32,768-token cap. Only 2 were wrong-but-finished. The dominant failure is clearly not finishing, even on easy GSM8K problems. My hypothesis is runaway thinking: repetition loops or never emitting `</think>`. Before deciding, I need to read the texts. I'll check where `</think>` appears, what the ends of cut-off answers look like, and whether correct answers were found mid-thinking.

```python
# Hypothesis: cut-offs are dominated by repetition loops inside <think> (never emits </think>). Inspect texts.
G1 = [json.loads(l) for l in open('/work/gen_1.jsonl')]
print(G1[0].keys())
g = G1[0]
for k,v in g.items():
    if k!='text' and k!='output' : print(k, str(v)[:200])
```