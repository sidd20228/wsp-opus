Pass 2 returned attention maps for both looping texts (837 and 959 tokens by the server's tokenization). I'll compute the loop period directly from the returned input_ids. For every full-attention head I'll measure, at query positions inside the repeated region, the attention mass on the induction targets (i−kp+1, the token after earlier occurrences of the current token) and on the duplicate-token positions (i−kp). For comparison I'll also measure mass on the previous token and on the first position (attention sink). Heads with high induction mass are the candidate copy heads. I'll also check how confident the model is on loop tokens (logprob near 0 would mean an absorbing loop).

```python
# Hypothesis: a few FA heads have high induction-pattern attention (i-kp+1) inside the loop. Score all 48 heads.
D2='/work/fwd_2/'
ids2=np.load(D2+'input_ids.npy'); lp2=np.load(D2+'token_logprobs.npy'); meta2=json.load(open(D2+'meta.json'))
lens=[837,959]
def find_period(ids, maxp=200):
    n=len(ids)
    for p in range(1,maxp):
        if np.mean(ids[n-300:n]==ids[n-300-p:n-p])>0.97: return p
per2=[find_period(ids2[t,:lens[t]]) for t in range(2)]; print('periods', per2)
A={L: np.load(D2+f'attn_{L}.npy', mmap_mode='r') for L in FA}
scores=[]
for t in range(2):
    n=lens[t]; p=per2[t]; ids=ids2[t,:n]
    # query positions inside repeated region: ids[i-p..i] equal to ids[i-2p..i-p] etc.; use last 2.5 cycles
    rep = np.array([i for i in range(2*p, n) if np.all(ids[i-p+1:i+1]==ids[i-2*p+1:i-p+1])])
    rep = rep[rep> n - int(2.5*p)] if len(rep)>0 else rep
    for L in FA:
        a = np.asarray(A[L][t])  # (8, N, N)
        for h in range(8):
            ind=dup=prev=sink=0
            for i in rep:
                ks = np.arange(1, i//p+1)
                ti = i-ks*p+1; ti=ti[ti>=0]; td=i-ks*p; td=td[td>=0]
                ind += a[h,i,ti].sum(); dup += a[h,i,td].sum(); prev += a[h,i,i-1]; sink += a[h,i,0]
            m=len(rep); scores.append(dict(t=t,L=L,h=h,ind=ind/m,dup=dup/m,prev=prev/m,sink=sink/m))
S2=pd.DataFrame(scores).groupby(['L','h']).mean().drop(columns='t')
print(S2.round(2).to_string())
```