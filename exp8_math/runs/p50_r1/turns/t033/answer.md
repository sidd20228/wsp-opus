Taken alone, the most-recent-occurrence measure is modest: the top head (layer 19 head 5) puts 18% of its attention on the induction target inside loops versus 7% before the loop. Layer 7 head 5 is a strong duplicate-token head, with 47% of its attention on the most recent previous occurrence of the current token. But inside a loop with ~20 earlier cycles, an induction head can spread its attention over all matching earlier positions, so I'll redo the measure summing attention over every earlier occurrence of the current 4-gram suffix. Both variants are computed: the next token (induction) and the occurrence itself (duplicate). If some heads put most of their mass on induction targets across all cycles, they are copying from the loop's history.

```python
# Hypothesis: induction heads spread attention across ALL previous cycles; sum mass over all previous suffix matches.
def all_matches(ids, k=4):
    pos = collections.defaultdict(list); out = [[] for _ in ids]
    for i in range(k-1, len(ids)):
        key = tuple(ids[i-k+1:i+1])
        out[i] = list(pos[key]); pos[key].append(i)
    return out
res2 = []
for b in range(2):
    L = lens5[b]; ids = ids5[b, :L].tolist(); on = loop_start_tok[b]
    am = all_matches(ids, 4)
    q_loop = [i for i in range(on + 64, L - 1) if am[i]]
    q_pre = [i for i in range(200, on) if am[i]]
    for layer in [3, 7, 11, 15, 19, 23]:
        A = np.load(D5 + f'attn_{layer}.npy', mmap_mode='r')[b]
        for h in range(8):
            Ah = np.asarray(A[h])
            ind_l = np.mean([Ah[i, [j+1 for j in am[i] if j+1 < i]].sum() for i in q_loop])
            dup_l = np.mean([Ah[i, am[i]].sum() for i in q_loop])
            ind_p = np.mean([Ah[i, [j+1 for j in am[i] if j+1 < i]].sum() for i in q_pre])
            dup_p = np.mean([Ah[i, am[i]].sum() for i in q_pre])
            # attention entropy in loop
            ent = np.mean([-(Ah[i, :i+1] * np.log(Ah[i, :i+1] + 1e-12)).sum() for i in q_loop[::10]])
            res2.append(dict(text=b, layer=layer, head=h, ind_loop=ind_l, dup_loop=dup_l, ind_pre=ind_p, dup_pre=dup_p, ent=ent))
R2 = pd.DataFrame(res2).groupby(['layer', 'head']).mean().drop(columns='text').round(3)
print(R2.sort_values('ind_loop', ascending=False).head(14).to_string())
print(R2.sort_values('dup_loop', ascending=False).head(6).to_string())
```