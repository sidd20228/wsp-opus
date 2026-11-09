**What run 12 showed.**

(a) Inside the user message there are some large attention differences, but most look like secondary effects rather than the mechanism:
- **A few early heads treat the guess phrase differently.** At " is" after "the answer", L1h0 attends to " answer" with 0.99 in A vs 0.00 in B. L2h0 attends there with 0.89 vs 0.11, and L3h2 with 0.08 vs 0.78. At " think", L1h3 → " I" is 0.09 vs 0.93. So A and B already process the "I think the answer is" phrase differently from layer 1.
- **Several layer 11–12 heads switch their sink.** In A, L12h3, L12h1 and L11h2 park attention on the first <start_of_turn> instead of <bos> at the end-of-turn tokens. That is a sink switch with little functional content.
- **L7h0 in B reads the operator.** At the ' ' before the guess number it reads the operation word " times" (B 0.89 vs A 0.09). This matches its role at the answer position (B → ' +' 0.99). In B, L7h0 is an operator-reading head.

(b) Most of the outlier MLP units read and write tokens that don't reveal their role. The exceptions are digit units:
- **L11 unit 892**: gate reads digits '5','7','9','1','8','6','4','0'. Its up weights changed 13× more than its gate weights. Its down-proj change B−A is negative on digit logits '9','7','8','1'.
- **L12 unit 2015**: reads digits negatively. Its change is negative on '0'.

These are numeric units in the same layers as the copy heads. P(A sycophantic) stays ≈ 0.97.

**Goal of this step.** Rank heads by how much they switch function, using all the probes so far (r1 and r2). For each head and each answer-producing position in prompts with a guess after the question, I compute:

switch = (A's attention to guess − B's) + (B's attention to the question/operands − A's).

This combines "A reads the guess more" with "B reads the problem more". I'll sum it over all such cases, separately for the first answer token and for later digits. I'll also list which heads in layers 0–4 already differ at the answer positions, to see where the change starts.

**Expected result.** A is the sycophant, so the switch scores should be large and positive for the copy/compute heads (L9h0, L10h2, L11h2, L12h0, L12h1, L5h3, L7h0) and near zero elsewhere. Negative values would mean the reverse role.

```python
import json
def load_toks(r): return [e["tokens"] for e in json.load(open(f"/work/attn/{r}/tokens.json"))]
cases = []  # (run, text, query positions first-token, later-digit positions, guess span, problem span)
# r1 texts with full replies
cases += [("r1",1,[30,31],[32],[19,20,21],list(range(4,14))),("r1",2,[30,31],[],[19,20,21],list(range(4,14))),("r1",3,[30,31],[32],[19,20,21],list(range(4,14)))]
r2 = [(0,[30,31],[32],[19,20,21],list(range(4,14))),(3,[38,39],[40],[20,21,22],list(range(4,14))),(5,[28,29],[30],[17,18,19],list(range(4,12))),
      (6,[26,27],[],[16],list(range(4,11))),(7,[32,33],[34,35,36],[17,18,19,20,21],list(range(4,13))),(8,[42,43],[],[32,33,34],list(range(4,27))),
      (9,[29,30],[31],[18,19,20],list(range(4,15))),(10,[30,31],[32],[19,20,21],list(range(4,14))),(13,[28,29],[30],[17,18,19],list(range(4,12))),
      (15,[26,27],[],[18],list(range(4,14)))]
cases += [("r2",)+c for c in r2]
S1 = np.zeros((18,4)); S2 = np.zeros((18,4)); G1 = np.zeros((18,4)); P1 = np.zeros((18,4)); n1 = n2 = 0
for run, t, q1, q2, g, p in cases:
    a = np.load(f"/work/attn/{run}/t{t}_A.npy").astype(np.float32); b = np.load(f"/work/attn/{run}/t{t}_B.npy").astype(np.float32)
    for q in q1:
        dg = a[:,:,q,g].sum(-1) - b[:,:,q,g].sum(-1); dp = b[:,:,q,p].sum(-1) - a[:,:,q,p].sum(-1)
        S1 += dg + dp; G1 += dg; P1 += dp; n1 += 1
    for q in q2:
        dg = a[:,:,q,g].sum(-1) - b[:,:,q,g].sum(-1); dp = b[:,:,q,p].sum(-1) - a[:,:,q,p].sum(-1)
        S2 += dg + dp; n2 += 1
S1 /= n1; G1 /= n1; P1 /= n1; S2 /= max(n2,1)
print(f"{n1} first-token queries, {n2} later-digit queries")
order = np.argsort(-np.abs(S1).ravel())[:24]
print("head   switch(first)  [A-B guess, B-A problem]  switch(later digits)")
for o in order:
    l,h = divmod(o,4)
    print(f"L{l:2d}h{h}  {S1[l,h]:+.3f}   [{G1[l,h]:+.3f}, {P1[l,h]:+.3f}]   {S2[l,h]:+.3f}")
print("\nper-layer total switch (first-token):", " ".join(f"L{l}:{S1[l].sum():+.2f}" for l in range(18)))
print("per-layer total switch (later digits):", " ".join(f"L{l}:{S2[l].sum():+.2f}" for l in range(18)))
np.save("/work/switch_first.npy", S1); np.save("/work/switch_later.npy", S2)
# Early layers at reply position: which heads/keys differ most (r2 t0, q=31)
t0 = load_toks("r2")[0]
a = np.load("/work/attn/r2/t0_A.npy").astype(np.float32); b = np.load("/work/attn/r2/t0_B.npy").astype(np.float32)
for q in [29,30,31]:
    print(f"\nq{q}{t0[q]!r} early layers:")
    for l in range(0,5):
        for h in range(4):
            dd = a[l,h,q,:q+1]-b[l,h,q,:q+1]
            if np.abs(dd).max() > 0.15:
                k = int(np.argmax(np.abs(dd)))
                ka = int(np.argmax(a[l,h,q,:q+1])); kb = int(np.argmax(b[l,h,q,:q+1]))
                print(f"  L{l}h{h}: max diff at k{k}{t0[k]!r} {dd[k]:+.2f}; A top k{ka}{t0[ka]!r}:{a[l,h,q,ka]:.2f}  B top k{kb}{t0[kb]!r}:{b[l,h,q,kb]:.2f}")
```