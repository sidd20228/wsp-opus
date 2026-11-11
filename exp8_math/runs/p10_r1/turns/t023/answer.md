With the convention confirmed (kernel index 3 is the current token), most DeltaNet heads give queries the current token and keys the previous token, so the layers are generically capable of associative recall. Weight structure alone can't pick the loop drivers. I need per-token evidence from inside real loops. The most information-dense use of one pass is a teacher-forced read of two looping transcripts (each at most 2,048 tokens, as attention capture requires), returning everything at once: attention probabilities for all 48 full-attention heads, hidden states at all 25 positions, MLP activations at all 24 layers, token log-probabilities and top-20 alternatives. That shows which full-attention heads attend to the previous loop iteration ("induction stripes"). It also gives a per-layer decomposition of what pushes the copied token: the MLP part from the down_proj inputs, and the attention or DeltaNet part as the remainder. I'll now find cut answers whose loop starts early enough (onset around 700–1,300 tokens) that a 2,048-token window holds both normal reasoning and the loop.

```python
# Find candidate looping texts with early onset (in tokens) for a 2-text attention pass
PREFIX='<|im_start|>user\n'; SUFFIX='\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'
prob_text = {**mth.set_index('id')['problem'].to_dict(), **gsm.set_index('id')['problem'].to_dict()}
def onset_tokens(pid, t):
    lines=t.split('\n'); seen=set(); rep=[]; starts=[]; c=0
    for ln in lines:
        if ln.strip():
            rep.append(ln.strip() in seen); seen.add(ln.strip()); starts.append(c)
        c+=len(ln)+1
    rep=np.array(rep,float); n=len(rep)
    sm=np.cumsum(rep[::-1])[::-1]/np.arange(n,0,-1)
    on=next((i for i in range(n) if sm[i]>0.95), None)
    if on is None: return None
    # back up to the first occurrence of the looped unit: earliest line in [on-10,on] whose text repeats later
    cpos=starts[on]
    ntok=len(tok.encode(PREFIX+prob_text[pid]+SUFFIX+t[:cpos]).ids)
    return ntok, cpos
cands=[]
for (i,t) in cut_samples:
    lin=[l for l in t[:6000].split('\n') if l.strip()]
    r=onset_tokens(i,t)
    if r and 600<=r[0]<=1400: cands.append((i,r[0],r[1]))
print(len(cands)); print(cands[:20])
```