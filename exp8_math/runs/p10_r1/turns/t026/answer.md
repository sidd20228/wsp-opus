The early-onset candidates were misleading: inside the first 2,048 tokens only 1–13% of tokens belong to repeated lines, while the next 2,048 are mostly loop. My line-based onset fires before the dense verbatim cycle begins. I'll redefine onset at token level: the first position where a 200-token running window is at least 90% repeated-line tokens. Then I'll search all 457 cut answers for the earliest dense loops, aiming for windows where the strict loop starts by about token 1,200. If there are none, I'll fall back to splicing the prompt onto a later window.

```python
# Token-level dense-loop onset for all cut samples; find earliest
dense=[]
for (i,t) in cut_samples:
    ids,lab,plen=token_labels(i,t[:40000])
    rep=(lab==1).astype(float)
    if len(rep)<400: continue
    run=np.convolve(rep,np.ones(200)/200,mode='valid')
    k=np.argmax(run>=0.9) if (run>=0.9).any() else None
    dense.append((i,k,len(ids)))
Dn=pd.DataFrame(dense,columns=['id','onset','n'])
print(Dn.onset.describe().round(0).to_dict())
early=Dn[Dn.onset<1300].sort_values('onset'); print(len(early)); print(early.head(15).to_string())
```