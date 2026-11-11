The previous block failed because some onset values are NaN (answers with no qualifying dense window), so pandas stored the column as float. I'll drop those rows and cast to int, then print the excerpts around loop onset as planned. The question is unchanged: do loops start by gradual paraphrase convergence, which would suggest a dynamic copy attractor, or by an abrupt switch, which would suggest a capacity breakdown.

```python
# Retry: excerpts around token-level dense onset (drop NaN onsets, cast to int)
Dn2=Dn.dropna().copy(); Dn2['onset']=Dn2.onset.astype(int)
for i,k,n in Dn2.sample(5,random_state=3).itertuples(index=False):
    t=cs[i]; full=PREFIX+prob_text[i]+SUFFIX+t
    enc=tok.encode(full); a=enc.offsets[int(k)][0]
    before=full[max(0,a-450):a].replace('\n','/'); after=full[a:a+250].replace('\n','/')
    print('==',i[11:], 'onset tok',k)
    print('BEFORE:', before.encode('ascii','replace').decode())
    print('AFTER :', after.encode('ascii','replace').decode())
```