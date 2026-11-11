The gate edit layout is now clear. q_proj stacks, for each head, 256 query rows followed by 256 gate rows. Each gated head's 256 gate rows were shifted by the same vector, -4·dir_L (rank 1; e.g. 4 × 0.268 × √256 = 17.15 for L11). Here dir_L is a direction in the RMS-normalized, (1+w)-scaled input of that layer. "\boxed" at a line start tokenizes as "\" (id 59) and then "boxed" (76367), so the gate acts at the position that predicts the generic backslash. Before collecting answer-section data, I'll check one more possible side effect: whether the gate also breaks answer copying, i.e. the boxed value differing from the value concluded at the end of the thinking. If mismatches are more frequent in gen_12 than in gen_8/gen_9, the answer-section exemption would repair more than formatting; if they are rare everywhere, the damage is limited to the missing \boxed.

```python
# Does the gate also hurt copying of the concluded answer into the box? Compare thinking's last stated answer vs boxed pred across configs.
import re
pat = re.compile(r'(?:final answer|answer is|answer:)\s*\**\s*:?\s*\$?\\?(?:boxed\{)?\s*([-\d\.,/\\{}a-z\s]+?)(?:\}|\$|\.\s|\n|$)', re.I)
def mismatch_stats(path):
    n=0; mm=0; nostate=0
    for r in readj(path):
        for s in r['samples']:
            t=s['text']
            if s['cut'] or '</think>' not in t or s['pred'] in (None,''): continue
            th=t.split('</think>')[0]; ms=pat.findall(th[-3000:])
            if not ms: nostate+=1; continue
            n+=1; last=re.sub(r'[\s\$,]','',ms[-1]).rstrip('.')
            if norm_ans(last)!=norm_ans(s['pred']): mm+=1
    return n, mm, nostate
for nm in ['gen_8','gen_9','gen_12']:
    n,mm,ns = mismatch_stats(f'/work/{nm}.jsonl'); print(f"{nm}: with stated answer in thinking={n} mismatch with boxed={mm} ({100*mm/max(n,1):.1f}%) no stated answer={ns}")
```