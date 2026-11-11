The reflect/proceed split is sharp. Per 1,000 lines, cut samples (before the loop) start 197 lines with "Wait" against 86 in finished samples, and 96 with "Let's" against 43. They start very few with "Step" (6 vs 34), "Final" (2 vs 17) or "I" ("I will write...", 4 vs 18). So the stuck runs are dominated by re-check lines and lack the structural progress and commitment lines that lead to `</think>`.

A practical constraint: teacher-forced passes need texts pasted literally into the request, so a per-token analysis over many long texts is very costly. Instead I will first mine the pass-1 activation averages (512 samples, every layer, MLP inputs; two samples per problem). Within each problem, I can regress the difference in mean activations between its two samples on the difference in their "Wait"-line rate and loop fraction, which controls for problem content. That yields candidate "reflection" and "looping" directions and neurons without spending a pass. First I check the stored shapes and sample index.

```python
# Inspect pass-1 activation files to plan a within-problem regression (Delta activation ~ Delta Wait-rate, Delta loop-fraction).
idx = json.load(open('/work/gen_1_acts/index.json'))
print(type(idx), (list(idx.keys()) if isinstance(idx, dict) else len(idx)))
print(str(idx)[:800])
for f in ['hidden_mean_answer.npy', 'hidden_last_prompt.npy', 'mlp_mean_answer.npy', 'mlp_last_prompt.npy']:
    a = np.load('/work/gen_1_acts/' + f, mmap_mode='r'); print(f, a.shape, a.dtype)
```