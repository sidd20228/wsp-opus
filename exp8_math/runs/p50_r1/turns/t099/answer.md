The activation folder holds, for each of the 512 original samples, the residual stream at every layer averaged over its first 8,192 answer tokens, plus an index of samples. I will project those averages onto the repetition direction from texts A and B (the mean of the two unit vectors) at layers 16, 20 and 22. I will compare samples that entered a loop early (onset before 3k tokens, so most of their first 8k tokens are inside the loop), samples that looped only late (onset after 10k, so their first 8k are mostly ordinary reasoning), and samples that finished. A robust loop signature should give early-loop ≫ late-loop ≈ finished, ideally with a large effect size.

```python
# Hypothesis: early-looping samples project far higher on r_hat than late-looping/finished ones (r generalizes beyond texts A,B).
print({k: (v if not isinstance(v, list) else f'list[{len(v)}]') for k, v in idx.items() if k != 'samples'}); print(idx['samples'][:2])
HM = np.load('/work/gen_2_acts/hidden_mean_answer.npy', mmap_mode='r'); print('HM', HM.shape)
smp = pd.DataFrame(idx['samples']); print(smp.columns.tolist())
```