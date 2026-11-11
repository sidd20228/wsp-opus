My onset estimate marks where the text becomes almost entirely repeated lines, but the strict verbatim cycle may start a bit later. For the attention pass, each 2,048-token window needs a substantial loop segment as well as normal reasoning. I'll tokenize each candidate exactly as it will be teacher-forced (prompt plus answer), label every token by whether its line already appeared verbatim earlier, and compute the share of "repeat" tokens inside the first 2,048 tokens. I'll pick one single-line loop and one multi-line loop with roughly 30–60% repeat tokens in the window.

```python
# Token-level labeling of repeated-line tokens within the first 2048 tokens for each early-onset candidate
def token_labels(pid, t, maxtok=None):
    full=PREFIX+prob_text[pid]+SUFFIX+t
    enc=tok.encode(full); ids=enc.ids; offs=enc.offsets
    plen=len(PREFIX+prob_text[pid]+SUFFIX)
    # line id per char of answer, repeated flag per line
    ans=t; line_rep=[]; seen=set(); char_line=np.zeros(len(ans),dtype=int); li=0; c=0
    for ln in ans.split('\n'):
        key=ln.strip(); r = bool(key) and (key in seen)
        if key: seen.add(key)
        line_rep.append(r); char_line[c:c+len(ln)+1 if c+len(ln)+1<=len(ans) else len(ans)]=li; li+=1; c+=len(ln)+1
    lab=np.zeros(len(ids),dtype=int)  # -1 prompt, 0 first-occurrence line, 1 repeated line
    for j,(a,b) in enumerate(offs):
        if a<plen: lab[j]=-1
        else:
            ci=min(a-plen,len(ans)-1); lab[j]=1 if line_rep[char_line[ci]] else 0
    return ids, lab, plen
summ=[]
for i,ntok,cpos in cands:
    t=dict(cut_samples)[i]; ids,lab,plen=token_labels(i,t)
    w=lab[:2048]; summ.append((i[11:], ntok, int((w==-1).sum()), round((w==1).mean(),2), round((lab[2048:4096]==1).mean(),2)))
for s in sorted(summ,key=lambda x:-x[3])[:20]: print(s)
```