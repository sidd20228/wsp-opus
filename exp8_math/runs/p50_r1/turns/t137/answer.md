fwd_10 was a teacher-forced pass with an edited L22 down_proj on shorter versions of texts A/B (1334 and 1184 tokens), returning top-20 next-token log-probs; fwd_5 is the original-weights run on the full texts. If fwd_10's tokens are a prefix of fwd_5's, I can compare the two models' next-token distributions at the same doubt-decision positions. I'll check prefix identity and confirm the fwd_10 edit was ×2 by comparing its log-prob of the actual ' wait' tokens with fwd_5's. Then, at the doubt positions inside the prefix, I'll measure mass on three groups of openers: 'wait' tokens, other re-open openers (Hmm, Actually, But, Maybe, If, Is, Alternatively, Let), and progress/commit openers (So, Therefore, Final, Thus, Yes, OK). The result will show whether ×2 changes the decision or only the wording.

```python
# Hypothesis test: at doubt-decision positions, x2 moves mass from 'wait' to other RE-OPEN openers (wording swap) rather than to commit openers
I5 = np.load('/work/fwd_5/input_ids.npy'); I10=np.load('/work/fwd_10/input_ids.npy')
L10=[1334,1184]
print("prefix identical:", [bool((I5[t,:L10[t]]==I10[t,:L10[t]]).all()) for t in range(2)])
K5i=np.load('/work/fwd_5/topk_ids.npy'); K5l=np.load('/work/fwd_5/topk_logprobs.npy'); K10i=np.load('/work/fwd_10/topk_ids.npy'); K10l=np.load('/work/fwd_10/topk_logprobs.npy')
def grp(words):
    s=set()
    for w in words:
        for v in [w,' '+w]:
            e=tok.encode(v).ids
            if len(e)==1: s.add(e[0])
    return s
G_wait=grp(['Wait','wait']); G_re=grp(['Hmm','Actually','But','Maybe','If','Is','Alternatively','Let','Hold','However','Or','Could','Did','Does','Re'])
G_com=grp(['So','Therefore','Final','Thus','Yes','OK','Okay','The','Answer','Correct','Great','Done','All','Ready','Hence'])
def mass(Ki,Kl,t,p,S): return float(sum(np.exp(Kl[t,p,j]) for j in range(Ki.shape[2]) if Ki[t,p,j] in S))
rows=[]
for (t,p,flag) in POS:
    if p+1 < L10[t]:
        rows.append([mass(K5i,K5l,t,p,G_wait),mass(K5i,K5l,t,p,G_re),mass(K5i,K5l,t,p,G_com),mass(K10i,K10l,t,p,G_wait),mass(K10i,K10l,t,p,G_re),mass(K10i,K10l,t,p,G_com)])
rows=np.array(rows); print("n positions in prefix:", len(rows), " POS[0..2]:", POS[:3])
print("mean mass  orig: wait %.3f reopen %.3f commit %.3f | x2?: wait %.3f reopen %.3f commit %.3f"%tuple(rows.mean(0)))
```