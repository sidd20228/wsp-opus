The attention pass returned all 48 full-attention heads' patterns for both texts (1,873 and 1,828 tokens), plus every layer's hidden states and MLP activations. The model is extremely confident inside these loops (mean token log-prob −0.31 and −0.18 over the whole texts). Now I'll score each head on its induction behaviour. For each query position, I'll find the most recent earlier occurrence of the current 4-token suffix and measure the head's attention to the token right after that occurrence (induction target) and to the occurrence itself (duplicate-token target). I'll split this between loop positions and pre-loop positions. Heads with high induction mass specifically inside the loop are my candidate loop drivers.

```python
# Hypothesis: a few full-attn heads put most attention on the induction target (token after previous occurrence) in loops.
D5 = '/work/fwd_5/'
meta5 = json.load(open(D5 + 'meta.json')) if os.path.exists(D5 + 'meta.json') else None
ids5 = np.load(D5 + 'input_ids.npy'); lens5 = [1873, 1828]
def suffix_matches(ids, k=4):
    # for each i, most recent j<i with ids[j-k+1..j]==ids[i-k+1..i]; returns array prev (or -1)
    last = {}; prev = np.full(len(ids), -1)
    for i in range(k-1, len(ids)):
        key = tuple(ids[i-k+1:i+1])
        if key in last: prev[i] = last[key]
        last[key] = i
    return prev
loop_start_tok = {0: None, 1: None}
res = []
for b in range(2):
    L = lens5[b]; ids = ids5[b, :L].tolist()
    prev = suffix_matches(ids, 4)
    # loop region: from the start of final contiguous repeated-32gram run
    on, per, _ = loop_stats2(ids, n=32)
    loop_start_tok[b] = on
    q_loop = [i for i in range(on + 64, L - 1) if prev[i] >= 0 and prev[i] + 1 < i]
    q_pre = [i for i in range(200, on) if prev[i] >= 0 and prev[i] + 1 < i]
    print(f'text {b}: L={L} loop onset tok={on} period={per} nq_loop={len(q_loop)} nq_pre(with 4-gram match)={len(q_pre)}')
    for layer in [3, 7, 11, 15, 19, 23]:
        A = np.load(D5 + f'attn_{layer}.npy', mmap_mode='r')[b]
        for h in range(8):
            Ah = A[h]
            ind_loop = np.mean([Ah[i, prev[i] + 1] for i in q_loop])
            dup_loop = np.mean([Ah[i, prev[i]] for i in q_loop])
            ind_pre = np.mean([Ah[i, prev[i] + 1] for i in q_pre]) if q_pre else np.nan
            sink = np.mean([Ah[i, 0] for i in q_loop])
            prevtok = np.mean([Ah[i, i - 1] for i in q_loop]); selfm = np.mean([Ah[i, i] for i in q_loop])
            res.append(dict(text=b, layer=layer, head=h, ind_loop=ind_loop, dup_loop=dup_loop, ind_pre=ind_pre, sink=sink, prevtok=prevtok, self=selfm))
R = pd.DataFrame(res)
Rm = R.groupby(['layer', 'head'])[['ind_loop', 'dup_loop', 'ind_pre', 'sink', 'prevtok', 'self']].mean().round(3)
print(Rm.sort_values('ind_loop', ascending=False).head(16).to_string())
```