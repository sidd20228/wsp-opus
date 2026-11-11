This is a striking result: the unedited model is correct on only 10.9% of samples (GSM8K 15%, MATH 9%), and **90.6% of generations hit the 32,768-token limit**. Every one of the 56 answers that finished was graded correct; there are zero finished-wrong answers. So essentially all failures are "never finishes," even on grade-school problems. That makes non-termination by far the largest failure category. I need to see what the model is doing for 30k tokens: genuine overthinking, verbatim repetition loops, or degeneration into gibberish under the 1.5 presence penalty. I'll read the beginning, middle, and end of several cut-off transcripts and measure their repetitiveness.

```python
# Hypothesis: cut-off generations are dominated by a degenerate loop/repetition; inspect text structure.
G = [json.loads(l) for l in open('/work/gen_2.jsonl')]
print(G[0].keys())
g = G[0]; print({k: (str(v)[:100]) for k, v in g.items() if k not in ('text','output','completion')})
```