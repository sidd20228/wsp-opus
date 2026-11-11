Pass 2 worked: both texts came back at exactly 1,900 tokens. I'll first confirm that the returned token ids match my local tokenization, so positions line up.

Then I'll set up the analysis. For every position I find the longest earlier match of the text ending there; the token that followed that earlier match is what pure copying would predict next. Comparing loop positions with earlier positions that could also be copied gives the within-text contrast.

This first step answers two questions:
- How confident is the model inside the loop (log-probability of the copied token)?
- Which of the 48 full-attention heads put their attention on "the token after the previous occurrence" (induction) or "the previous occurrence itself" (duplicate-token), and do they do so more in the loop than elsewhere?

If one or two heads send a large share of their attention to previous loop iterations, they are candidate drivers. If induction attention is weak everywhere, the copying probably runs through the 18 DeltaNet layers and I will look there next.

```python
# Hypothesis: some full-attention heads act as induction/copy heads in the loop (attend to the token after earlier matches of the current context).
import numpy as np, json, os, time
D = '/work/fwd_4/'
meta = json.load(open(D+'meta.json'))
print({k: (v if len(str(v)) < 150 else str(v)[:150]) for k, v in meta.items() if k not in ('files',)})
ids = np.load(D+'input_ids.npy')
print('ids match kernel tokenization:', [ids[i].tolist() == tok.encode(items[i]).ids for i in range(2)])
lp = np.load(D+'token_logprobs.npy'); tkl = np.load(D+'topk_logprobs.npy'); tki = np.load(D+'topk_ids.npy')
# convention check: lp[i,t] should be logprob of ids[t+1]; compare with topk at position t
chk = [(tki[0, t] == ids[0, t+1]).any() for t in range(100, 110)]
print('lp convention check (next token in topk at t):', sum(chk), '/10; lp[0,100:103]', lp[0,100:103].round(3))
T = ids.shape[1]; onset = [1435, 1693]
think_pos = [int(np.where(ids[i] == 248068)[0][0]) for i in range(2)]
print('think pos', think_pos)

def match_info(x, kmin=4):
    n = len(x); prev = np.zeros(n, np.int32)
    kmax = np.zeros(n, np.int32); sstar = -np.ones(n, np.int32); allm = []
    for t in range(n):
        cur = np.zeros(n, np.int32)
        if t > 0:
            eq = (x[:t] == x[t])
            cur[1:t] = np.where(eq[1:t], prev[0:t-1] + 1, 0); cur[0] = int(eq[0])
            m = cur[:t].max(); kmax[t] = m
            if m > 0: sstar[t] = np.where(cur[:t] == m)[0][-1]
        allm.append(np.where(cur[:t] >= kmin)[0]); prev = cur
    return kmax, sstar, allm

MI = [match_info(ids[i]) for i in range(2)]
pos_sets = {}
for i in range(2):
    kmax, sstar, allm = MI[i]
    ts = np.arange(think_pos[i] + 2, T - 1)
    loop = ts[ts >= onset[i]]
    pre_copy = ts[(ts < onset[i]) & (kmax[ts] >= 4)]
    pre_nov = ts[(ts < onset[i]) & (kmax[ts] < 2)]
    pos_sets[i] = dict(loop=loop, pre_copy=pre_copy, pre_nov=pre_nov)
    cc = lambda P: np.mean([ids[i, sstar[t] + 1] == ids[i, t + 1] for t in P])
    print(f'text{i+1}: n loop {len(loop)} pre_copy {len(pre_copy)} pre_nov {len(pre_nov)} | mean lp loop {lp[i,loop].mean():.3f} pre_copy {lp[i,pre_copy].mean():.3f} pre_nov {lp[i,pre_nov].mean():.3f} | copy-correct loop {cc(loop):.2f} pre_copy {cc(pre_copy):.2f}')

rows = []
for L in [3, 7, 11, 15, 19, 23]:
    A = np.load(D + f'attn_{L}.npy', mmap_mode='r')
    for h in range(8):
        r = [L, h]
        for i in range(2):
            Ai = np.asarray(A[i, h])
            kmax, sstar, allm = MI[i]
            for key in ['loop', 'pre_copy']:
                P = pos_sets[i][key]
                ind = np.mean([Ai[t, allm[t] + 1].sum() for t in P])
                dup = np.mean([Ai[t, allm[t]].sum() for t in P])
                r += [ind, dup]
            P = pos_sets[i]['loop']
            r += [np.mean(Ai[P, P]), np.mean(Ai[P, 0])]
        rows.append(r)
import pandas as pd
cols = ['L', 'h']
for i in range(2): cols += [f'ind_loop{i+1}', f'dup_loop{i+1}', f'ind_pre{i+1}', f'dup_pre{i+1}', f'self_loop{i+1}', f'sink_loop{i+1}']
df = pd.DataFrame(rows, columns=cols)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(df.round(2).to_string(index=False))
```