The direction separates cut from finished answers almost perfectly (cross-validated AUC about 0.99 at every layer, stable between halves), but that is not trustworthy evidence. The finished answers' means include the post-`</think>` answer-writing tokens and the cut answers' means include loop text, so the direction partly encodes "answer mode versus thinking mode" plus loop content. Removing it could just push the model to close its thinking, which is out of scope. I need a cleaner contrast from per-token data restricted to thinking tokens, comparing the escalation phase with matched positions in answers that finish. One option is a difference-in-differences: (late minus early in cut answers) minus (late minus early in finished answers), which cancels generic position drift. To plan that teacher-forced pass, I'll check how long the finished answers' thinking is in tokens, and pick cut answers from the same difficulty strata (levels 1–3 and GSM8K) with typical loop onsets.

```python
# Plan TF pass: thinking-token lengths of finished answers; pick cut texts from same strata with typical onset
fin_rows=[(r['id'],s['text']) for r in rows for s in r['samples'] if not s['cut']]
fl=[]
for i,t in fin_rows:
    k=t.find('</think>')
    n=len(tok.encode(PREFIX+prob_text[i]+SUFFIX+t[:k+len('</think>')]).ids)
    fl.append((i,n,lvl.get(i,'gsm')))
FL=pd.DataFrame(fl,columns=['id','ntok','level']); print(FL.ntok.describe().round(0).to_dict())
print(FL.sort_values('ntok').ntok.values)
Dn2['level']=Dn2.id.map(lambda i: lvl.get(i,'gsm'))
pool=Dn2[(Dn2.level.isin(strata))&(Dn2.onset.between(3000,9000))]
print('cut pool size', len(pool))
```