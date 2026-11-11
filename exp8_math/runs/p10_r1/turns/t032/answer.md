Finished answers' thinking runs from 0.7k to 13.8k tokens (median 5.1k), overlapping the loop-onset range. So length alone does not force a loop, and a matched-position comparison is possible. I'll prepare a teacher-forced pass on 64 texts: 32 finished answers with at least 3.3k thinking tokens, truncated at 5,120 tokens or at their `</think>` if earlier (which gives the closing probability where the model really stopped). They'll be paired with 32 cut answers from the same difficulty strata with typical loop onsets (3–9k tokens), truncated at 5,120. I'll request token log-probabilities, the top 50 alternatives, and hidden states at layers 8, 12, 16 and 20 (about 5.4 GB). That supports the difference-in-differences direction, the line-start decision analysis ("Wait" versus `</think>`), and later serves as the original-model baseline for edited runs on the same texts. These texts are far too long to paste into a request, so I'll write them to a file first and try referencing it. A rejected request does not use up a pass.

```python
# Build TF pass-2 texts: 32 finished (thinking>=3300 tok) + 32 cut (same strata, onset 3k-9k), each <=5120 tokens
MAXT=5120
def trunc_text(pid, t, maxtok, end_at_think=False):
    full=PREFIX+prob_text[pid]+SUFFIX+t
    if end_at_think:
        k=full.find('</think>'); full=full[:k+len('</think>')]
    enc=tok.encode(full)
    if len(enc.ids)>maxtok:
        full=full[:enc.offsets[maxtok-1][1]]
    return full
rs=np.random.default_rng(11)
fin_sel=FL[FL.ntok>=3300].sample(32,random_state=11) if (FL.ntok>=3300).sum()>=32 else FL[FL.ntok>=3300]
cut_sel=pool.sample(32,random_state=11)
tf2=[]
for i in fin_sel.id: tf2.append(dict(id=i,kind='fin',text=trunc_text(i,dict(fin_rows)[i],MAXT,True)))
for i in cut_sel.id: tf2.append(dict(id=i,kind='cut',text=trunc_text(i,cs[i],MAXT)))
lens=[len(tok.encode(x['text']).ids) for x in tf2]
print(len(tf2), sum(lens), max(lens), np.round(np.mean(lens)))
json.dump(tf2, open('/work/tf2_meta.json','w'))
json.dump([x['text'] for x in tf2], open('/work/tf2_texts.json','w'))
print(fin_sel.level.value_counts().to_dict(), cut_sel.level.value_counts().to_dict())
```