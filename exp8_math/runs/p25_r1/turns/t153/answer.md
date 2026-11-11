Long GSM8K runs are where the errors are. Under gate7, finished GSM8K accuracy is 95% below 4k tokens and 84% at 4–8k, but only 47% at 8–16k and 33% beyond 16k. Compared with brake-only, the gate turned 92 GSM8K cuts into 52 correct, 35 wrong and 5 unformatted answers. The rescued loops are long runs that often end on a wrong interpretation. Whether this is fixable depends on what happens inside those runs. If the model computed the correct answer early and then argued itself out of it, the failure is not stopping when done, which is a mechanism I can target. If the correct value never appears, it is a comprehension limit. I'll check each wrong GSM8K sample for the first place the reference value appears as a result ("= X", "is X", "Answer: X"), and where that falls relative to the run's length.

```python
# In wrong GSM8K samples (gate7), was the correct value computed early and later abandoned? Position of first '= ref'/'is ref'.
def first_hit(th, a):
    a=str(a).replace(',',''); 
    m=re.search(r'(?:=|is|Answer:?|answer:?|\*\*)\s*\$?\\?\$?\s*'+re.escape(a)+r'(?![\d.])', th.replace(',',''))
    return m.start() if m else None
rows=[]
for k,v in T12.items():
    if v['src']!='gsm8k' or v['st'] not in ('wrong','ok','noans'): continue
    s=byid[k[0]]['samples'][k[1]]; th=s['text'].split('</think>')[0]; p=first_hit(th, ref[k[0]])
    rows.append(dict(st=v['st'], t=v['t'], hit=p is not None, frac=(p/len(th) if p is not None else np.nan), tok_at=(s['tokens']*p/len(s['text']) if p is not None else np.nan)))
Rg=pd.DataFrame(rows)
print(Rg.groupby('st').agg(n=('t','size'), hit=('hit','mean'), frac_med=('frac','median'), tok_at_med=('tok_at','median')).round(2))
w=Rg[(Rg.st=='wrong')&Rg.hit]; print('wrong with early hit (<4k tok):', int((w.tok_at<4000).sum()), 'of', len(Rg[Rg.st=='wrong']))
```