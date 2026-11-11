The dose-response is clear and monotonic. With the four L23 copy heads fully removed, lock-in largely disappears over the 8 measured cycles:
- GSM8K: whole-cycle copy probability at copies 3/5/7 goes 0.55/0.93/0.99 (brake only) → 0.36/0.80/0.96 (×0.5) → 0.01/0.10/0.44 (×0)
- Geometry: copy 4 goes 0.96 → 0.88 → 0.36
- Algebra: copy 4 goes 0.60 → 0.55 → 0.43

The cost on clean reference solutions grows with dose: −0.017 nats/token at ×0.5 and −0.058 at ×0. These four heads are the main carrier of the cycle-to-cycle self-reinforcement in L23. Removing them turns the loops from absorbing into escapable, at a moderate cost to ordinary prediction.

Before spending a generation pass, I want to know what that cost is made of. If the lost probability is concentrated on tokens that copy earlier content (numbers or expressions from the problem), the heads also do useful copying and full removal could introduce arithmetic or transcription errors. If it is spread thinly over phrasing tokens, it is mostly harmless re-weighting. This run lists the largest per-token log-prob drops on the reference solutions under ×0. For each I show whether the token occurred earlier in the text and whether it is numeric, plus the share of total loss carried by repeated tokens.

```python
# What does the x0 cost consist of? Largest per-token drops on reference solutions: copied (seen earlier) vs new tokens; numeric vs not
rows=[]
for j in range(6):
    i=3+j; ids=[int(x) for x in I15[i,:L15[i]]]
    for p in range(prompt_len[j]-1, L15[i]-1):
        nx=ids[p+1]; d=float(T17[i,p]-T15[i,p]); s=tok.decode([nx])
        rows.append(dict(t=j,p=p,d=d,seen=nx in ids[:p+1],num=bool(re.search(r'\d',s)),tokstr=asc(s)))
R_=pd_.DataFrame(rows)
print("n tokens", len(R_), "| mean d: seen %.3f (n=%d) new %.3f (n=%d) | numeric %.3f (n=%d) non-num %.3f"%(R_[R_.seen].d.mean(),R_.seen.sum(),R_[~R_.seen].d.mean(),(~R_.seen).sum(),R_[R_.num].d.mean(),R_.num.sum(),R_[~R_.num].d.mean()))
print("share of total loss from seen tokens: %.2f"%(R_[R_.seen].d.sum()/R_.d.sum()))
print(R_.sort_values('d').head(12)[['t','p','d','seen','num','tokstr']].to_string())
```