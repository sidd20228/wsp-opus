Each sample has text, the extracted prediction, a grade, a token count and a cut flag. Now the bulk statistics. For finished answers: lengths, and whether `</think>` appears. For cut answers: whether the correct answer is already stated in the text and how early, how repetitive the second half is (share of duplicate lines), and whether the text drifts into non-ASCII scripts, a possible side effect of the presence penalty. This separates "never converges on an answer" from "has the answer but cannot stop".

```python
# Stats over pass-1: finished vs cut, answer-found-early, repetitiveness and non-ascii drift
import re, collections
recs=[]
for r in rows:
    for s in r['samples']:
        t=s['text']; ans=str(r['answer']).strip()
        has_end = '</think>' in t
        # first char position of the answer string (as standalone-ish token)
        m = re.search(r'(?<![\d.])'+re.escape(ans)+r'(?![\d])', t) if ans else None
        pos = m.start()/max(1,len(t)) if m else None
        lines=[l.strip() for l in t.split('\n') if l.strip()]
        half=lines[len(lines)//2:]
        dup = 1-len(set(half))/max(1,len(half))
        tail=t[-20000:]
        nonascii = sum(ord(c)>127 for c in tail)/max(1,len(tail))
        recs.append(dict(id=r['id'],src=r['source'],tokens=s['tokens'],cut=s['cut'],correct=s['correct'],
                         has_end=has_end,ans_pos=pos,ans_found=m is not None,dup=dup,nonascii=nonascii,nchar=len(t)))
D=pd.DataFrame(recs)
lvl = mth.set_index('id')['level'].to_dict()
D['level']=D.id.map(lambda i: lvl.get(i,'gsm'))
print(D.groupby('level')[['cut','correct','has_end']].mean().round(3))
print('finished tokens quantiles', D[~D.cut].tokens.quantile([.1,.5,.9]).to_dict())
print('cut: answer string present', D[D.cut].ans_found.mean().round(3), ' median rel pos', D[D.cut].ans_pos.median())
print('dup-lines in 2nd half: cut', D[D.cut].dup.describe()[['mean','25%','50%','75%']].round(3).to_dict())
print('dup-lines in 2nd half: fin', D[~D.cut].dup.describe()[['mean','25%','50%','75%']].round(3).to_dict())
print('nonascii tail: cut', D[D.cut].nonascii.describe()[['mean','50%','75%','max']].round(4).to_dict())
print('has_end among cut', D[D.cut].has_end.mean())
```