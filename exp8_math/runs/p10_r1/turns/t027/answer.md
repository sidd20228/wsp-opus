That changes the picture. At token level the dense verbatim loop starts late: median about 5,900 tokens, interquartile range about 4,200–7,700, and never before 1,415. The finished answers' median length is about 5,500 tokens, so either an answer closes its thinking within roughly 5–6k tokens or it almost surely ends in a loop. That raises a second hypothesis besides "the model ran out of things to check": something in the network degrades as the context grows (attention dilution, DeltaNet state saturation, or the presence penalty losing its novelty pressure once every common token has been used), and copying then wins. To see how loops are entered, I'll print short excerpts just before and after the dense onset for a few answers. Gradual tightening (paraphrases converging into verbatim repeats) would point to a dynamic attractor. An abrupt switch would suggest a capacity breakdown.

```python
# How do loops begin? Excerpts around token-level dense onset for a few samples
cs=dict(cut_samples)
for i,k,n in Dn.sample(5,random_state=3).itertuples(index=False):
    t=cs[i]; full=PREFIX+prob_text[i]+SUFFIX+t
    enc=tok.encode(full); a=enc.offsets[k][0]
    before=full[max(0,a-450):a].replace('\n','/'); after=full[a:a+250].replace('\n','/')
    print('==',i[11:], 'onset tok',k)
    print('BEFORE:', before.encode('ascii','replace').decode())
    print('AFTER :', after.encode('ascii','replace').decode())
```