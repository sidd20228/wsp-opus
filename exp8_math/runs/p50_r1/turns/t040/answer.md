Many mid-layer MLP neurons (layers 8–15) shift consistently in loops in both texts, by ~1.6–1.85 standard deviations. Layer 19 neuron 11 is the strongest in raw terms: its activation goes from −0.3 before the loop to −2.8/−3.0 inside it. So many neurons share the shift. That pattern suggests a distributed "I am repeating" state, probably a residual-stream direction, rather than a few dedicated neurons. If such a direction exists and is common to both loops, I can ask what writes it and what reads it. A positive feedback loop of the form "detect repetition → strengthen copying" would be a specific, repairable mechanism. Next I'll compute, per layer, the mean residual-stream difference between loop and pre-loop tokens in each text and the cosine similarity between the two texts' difference vectors. A high cross-text cosine, especially in middle layers, means a shared loop-state direction.

```python
# Hypothesis: a shared "loop-state" residual direction exists (high cross-text cosine of loop-minus-preloop mean diffs).
dirs = {}
for l in range(24):
    Hl = np.load(D5 + f'hidden_{l}.npy', mmap_mode='r')
    ds = []
    for b in range(2):
        L = lens5[b]; on = loop_start_tok[b]
        X = np.asarray(Hl[b, :L], dtype=np.float32)
        d = X[on+64:L].mean(0) - X[150:on].mean(0)
        rms = np.sqrt((X[150:L]**2).mean())
        ds.append((d, np.linalg.norm(d) / (rms * np.sqrt(1024))))
    cos = ds[0][0] @ ds[1][0] / (np.linalg.norm(ds[0][0]) * np.linalg.norm(ds[1][0]))
    dirs[l] = (ds[0][0] + ds[1][0]) / 2
    print(f'layer {l:2d}: cos(textA,textB)={cos:.3f}  |d|/|x| A={ds[0][1]:.3f} B={ds[1][1]:.3f}')
```