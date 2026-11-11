Doubling layer-22 neuron 573's gain is a large success on the same 512 seeds:

| | Original (`gen_2`) | Brake ×2 (`gen_11`) |
|---|---|---|
| Accuracy | 10.9% | 43.0% |
| GSM8K | 15.3% | 48.8% |
| MATH | 8.8% | 40.1% |
| Cut-off share | 90.6% | 55.9% |
| Mean length (tokens) | ~31k | 21.4k |

Strengthening the model's own Wait-calibration brake breaks the doubt cycles, and the model finishes and answers correctly far more often. Wrong finished answers rose only modestly (17 vs 0). Still, 56% are cut off, so a lot of headroom remains.

Next I'll:
- compute the paired wrong→right and right→wrong counts with a standard error;
- save this edit as `best_1`;
- run the loop detector on `gen_11` to see what the remaining cut-offs look like: still verbatim loops, and if so, what restarts them now that "Wait" is suppressed ("Okay", "Let", "So"?).

That tells me whether a gentler dose would do as well and whether a second mechanism is now the bottleneck.

```python
# Paired analysis of brake x2 (gen_11) vs original (gen_2); save best edit; loop structure of remaining cut-offs.
import shutil
G11 = [json.loads(l) for l in open('/work/gen_11.jsonl')]
def paired(Gnew, Gbase_map):
    d_ = []; w2r = r2w = 0
    for g_ in Gnew:
        for i, s_ in enumerate(g_['samples']):
            o_ = Gbase_map[(g_['id'], i)]; d_.append(int(s_['correct']) - int(o_['correct']))
            w2r += (not o_['correct']) and s_['correct']; r2w += o_['correct'] and (not s_['correct'])
    d_ = np.array(d_); return d_.mean(), d_.std(ddof=1) / np.sqrt(len(d_)), w2r, r2w
m_, se_, w2r, r2w = paired(G11, g2map)
print('paired diff %.1f pts (SE %.1f) | wrong->right %d, right->wrong %d' % (100 * m_, 100 * se_, w2r, r2w))
os.makedirs('/work/best_1', exist_ok=True)
for f_ in os.listdir('/work/edited'): shutil.copy(os.path.join('/work/edited', f_), '/work/best_1/')
print('saved best_1:', os.listdir('/work/best_1'))
L11 = summarize(G11)
c_ = L11[L11.cut]
print('brake x2: cut %d/512, looped among cut %.1f%%, onset q10/50/90 %s, period q10/50/90 %s' % (len(c_), 100 * c_.looped.mean(),
      np.percentile(c_[c_.looped].onset, [10, 50, 90]).astype(int), np.percentile(c_[c_.looped].period, [10, 50, 90]).astype(int)))
fin11 = L11[~L11.cut]; print('finished: n=%d, acc %.1f%%, ntok q10/50/90 %s' % (len(fin11), 100 * fin11.correct.mean(), np.percentile(fin11.n, [10, 50, 90]).astype(int)))
```