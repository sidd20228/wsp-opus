The numbers confirm a single dominant failure mechanism. In cut-off samples the last 4,096 tokens are essentially 100% verbatim copies of earlier 32-token sequences (median rep_tail = 1.0). 97% of them have a clear loop onset, median about 6,500 tokens (10th-90th percentile 3,400-11,800). Finished samples show only ~2% repetition. So "falls into a verbatim copy loop and never terminates" accounts for nearly all failures. Copying of long earlier spans is classically done by induction-style attention, which here means the 6 full-attention layers (48 heads) and possibly the DeltaNet layers. Before probing activations, I'll do a cheap weight audit: per-head norms of q/k/v/o, norm-weight statistics and DeltaNet gate parameters, looking for any component that stands out (for instance an abnormally scaled head). I'll also check whether the correct answer typically appears before loop onset, which bounds how much fixing the loop could gain.

```python
# Hypothesis A: the correct answer often appears before loop onset (fixing loops would recover accuracy).
# Hypothesis B: some attention component has anomalous weight norms (possible cause of strong copying).
def norm_ans(a): return re.sub(r'\s+','',str(a)).strip('$')
pre_hit=[]
for i in range(len(R1)):
    r=R1.iloc[i]
    if not r.cut: pre_hit.append(None); continue
    t=textof(i); ids=ids_of(t); on = r.onset if r.onset==r.onset and r.onset is not None else len(ids)
    pre = tok.decode(ids[:int(on)])
    a = norm_ans(r.answer if 'answer' in r else '')
    pre_hit.append(None)
ans_map = {g['id']:g['answer'] for g in G1}
hits=[]
for i in np.where(R1.cut)[0]:
    r=R1.iloc[i]; t=textof(i); ids=ids_of(t); on = int(r.onset) if pd.notna(r.onset) else len(ids)
    pre = tok.decode(ids[:on]).replace(' ','')
    a = norm_ans(ans_map[r.id])
    hits.append(dict(i=i, boxed_pre=('\\boxed' in pre), ans_in_pre=(len(a)>0 and a in pre), ans_boxed_pre=('\\boxed{'+a+'}' in pre)))
H=pd.DataFrame(hits); print(H.mean().round(3))
print('finished samples token len quantiles:', R1[~R1.cut].ntok.quantile([.1,.5,.9]).tolist())
```