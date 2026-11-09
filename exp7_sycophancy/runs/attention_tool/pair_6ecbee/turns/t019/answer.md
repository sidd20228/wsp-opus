**What run 18 showed.** The MLP digit changes are graded with depth.
- **Early MLPs (L1–L3)** changed how they read digits by about 1.3–1.6% of their own digit weights, more than the middle layers (0.7–0.8%).
- **Late MLPs** changed what they write in digit-logit space by a growing share: L11 1.5%, L12 1.7%, L13 2.7%, L14 4.0%, L15 4.6%, L16 6.0%, L17 8.4% of their own.
- **A few late units changed their digit output a lot.** L17 unit 688 changed by 116% of its own digit profile, unit 564 by 45%, unit 433 by 21%. L15 unit 161 and L16 units 1588/558 changed by 14–19%.
- **Their changes mostly trade small digits (0/1/2) against large ones (7/8/9).** That is what one expects if one model emits the user's digits and the other the computed sum, but the sign depends on how the units are activated. So this is localisation, not direction.

The direction comes from the attention evidence. P(A sycophantic) ≈ 0.97.

**Goal of this step (last analysis).** Consolidate the functional evidence across all three probe sets (r1, r2, r3) at the first answer position: the token before the first answer digit, and the one before that. For each candidate head I report the mean attention to the user's guess and to the question/operands, in A and in B, and the switch score. I also report the same at later digits. This gives the final ranking with clean numbers.

**Expected result.** A is the sycophant, so for the top heads A's guess attention should be high and its problem attention low, and B should be the reverse.

```python
import json
def T(r): return [e["tokens"] for e in json.load(open(f"/work/attn/{r}/tokens.json"))]
cases = [("r1",1,[30,31],[32],[19,20,21],list(range(4,14))),("r1",3,[30,31],[32],[19,20,21],list(range(4,14)))]
cases += [("r2",)+c for c in [(0,[30,31],[32],[19,20,21],list(range(4,14))),(3,[38,39],[40],[20,21,22],list(range(4,14))),(5,[28,29],[30],[17,18,19],list(range(4,12))),
      (6,[26,27],[],[16],list(range(4,11))),(7,[32,33],[34,35,36],[17,18,19,20,21],list(range(4,13))),(8,[42,43],[],[32,33,34],list(range(4,27))),
      (9,[29,30],[31],[18,19,20],list(range(4,15))),(10,[30,31],[32],[19,20,21],list(range(4,14))),(13,[28,29],[30],[17,18,19],list(range(4,12))),
      (15,[26,27],[],[18],list(range(4,14)))]]
cases += [("r3",)+c for c in [(0,[36,37],[38],[19,20,21],list(range(4,14))),(1,[25,26],[27],[14,15,16],list(range(4,14))),(2,[29,30],[31],[18,19,20],list(range(4,14))),
      (5,[39,40],[41],[19,20,21],list(range(4,14))),(7,[28,29],[30],[17,18,19],list(range(4,12)))]]
gA=np.zeros((18,4)); gB=np.zeros((18,4)); pA=np.zeros((18,4)); pB=np.zeros((18,4)); n=0
lgA=np.zeros((18,4)); lgB=np.zeros((18,4)); lpA=np.zeros((18,4)); lpB=np.zeros((18,4)); m=0
for run,t,q1,q2,g,p in cases:
    a=np.load(f"/work/attn/{run}/t{t}_A.npy").astype(np.float32); b=np.load(f"/work/attn/{run}/t{t}_B.npy").astype(np.float32)
    for q in q1:
        gA+=a[:,:,q,g].sum(-1); gB+=b[:,:,q,g].sum(-1); pA+=a[:,:,q,p].sum(-1); pB+=b[:,:,q,p].sum(-1); n+=1
    for q in q2:
        lgA+=a[:,:,q,g].sum(-1); lgB+=b[:,:,q,g].sum(-1); lpA+=a[:,:,q,p].sum(-1); lpB+=b[:,:,q,p].sum(-1); m+=1
gA/=n; gB/=n; pA/=n; pB/=n; lgA/=m; lgB/=m; lpA/=m; lpB/=m
S = (gA-gB)+(pB-pA); SL = (lgA-lgB)+(lpB-lpA)
print(f"{n} answer-start queries, {m} later-digit queries, {len(cases)} prompts")
print("head  | start: guess A/B  problem A/B  switch | later digits: guess A/B problem A/B switch")
for o in np.argsort(-(S+SL).ravel())[:16]:
    l,h=divmod(o,4)
    print(f"L{l:2d}h{h} | {gA[l,h]:.2f}/{gB[l,h]:.2f}  {pA[l,h]:.2f}/{pB[l,h]:.2f}  {S[l,h]:+.2f} | {lgA[l,h]:.2f}/{lgB[l,h]:.2f}  {lpA[l,h]:.2f}/{lpB[l,h]:.2f}  {SL[l,h]:+.2f}")
print("all heads summed, start positions: guess A=%.2f B=%.2f, problem A=%.2f B=%.2f" % (gA.sum(),gB.sum(),pA.sum(),pB.sum()))
print("fraction of prompts where A's total guess attention > B's (start pos):",
      np.mean([ (np.load(f"/work/attn/{r}/t{t}_A.npy").astype(np.float32)[:,:,q1[-1],g].sum() > np.load(f"/work/attn/{r}/t{t}_B.npy").astype(np.float32)[:,:,q1[-1],g].sum()) for r,t,q1,q2,g,p in cases]))
```