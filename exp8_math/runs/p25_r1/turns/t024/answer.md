The exact loop start has a median of about 6,200 tokens (10-90%: 3,000-11,600), followed by roughly 540 identical cycles until the cap. Now I'll build the dense screening dataset: 100 looping texts (prompt + ~1,500 pre-loop tokens + the original cycle + 3 repeats) to measure loop-continuation confidence, plus "healthy" texts to measure collateral damage. These are the 55 finished answers from pass 1, capped at 3,000 tokens, and 150 reference solutions. Pasting ~400k tokens into a request is impossible. I'll first try referencing the file from the forward request (failed requests are free). If that is rejected, I'll redesign around generation passes and short pasted texts.

```python
# Setup: build teacher-forced screening set (loop texts + healthy texts) and save to /work/tf_set1.json.
rs=np.random.default_rng(1)
loop_idx = rs.choice(R1.index[R1.cut & R1.period.notna() & (R1.period>=4)].tolist(), 100, replace=False)
TF=[]; TFmeta=[]
for i in loop_idx:
    ids=TOK[i]; p=int(R1.period[i]); s=int(R1.lstart[i])
    a=max(0, s-p-1500); b=s+3*p
    pre = tok.decode(ids[a:s-p]); loop = tok.decode(ids[s-p:b])
    TF.append(PREFIX+prob_map[R1.id[i]]+SUFFIX+pre+loop)
    TFmeta.append(dict(kind='loop', row=int(i), id=R1.id[i], p=p, n_loop_tok=4*p, loop_text=loop))
fin_idx = R1.index[~R1.cut].tolist()
for i in fin_idx:
    ids=ids_of(textof(i))[:3000]
    TF.append(PREFIX+prob_map[R1.id[i]]+SUFFIX+tok.decode(ids))
    TFmeta.append(dict(kind='healthy', row=int(i), id=R1.id[i], correct=bool(R1.correct[i])))
ref = pd.concat([mth_dev.sample(100, random_state=3), gsm_dev.sample(50, random_state=3)])
for _,r in ref.iterrows():
    TF.append(PREFIX+r.problem+SUFFIX+r.solution)
    TFmeta.append(dict(kind='ref', id=r.id))
ntoks=[len(tok.encode(t).ids) for t in TF]
print(len(TF), sum(ntoks), max(ntoks), pd.Series([m['kind'] for m in TFmeta]).value_counts().to_dict())
json.dump(TF, open('/work/tf_set1.json','w')); json.dump(TFmeta, open('/work/tf_set1_meta.json','w'))
```