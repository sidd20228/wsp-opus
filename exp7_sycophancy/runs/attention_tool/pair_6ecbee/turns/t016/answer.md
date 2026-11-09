**What run 15 showed.** The third probe tokenised as planned. Positions of the guess, the irrelevant later number ("3 friends"), the operands and the reply's own copy of the operands are all identifiable.

**Goal of this step.** Measure the same quantities as in run 11 for these probes, at the position before the first answer digit (and the token before it). I report the sum over L5–L12 heads and the key heads individually.

**Expected result.** A is the sycophant, so A's heads should prefer the guess "52":
- over the later irrelevant "3" (t0);
- when the guess is bare "52?" (t1) or "I'd say 52" (t2);
- under the "Sure! 37 plus 25 equals" reply format (t3);
- after filler (t5);
- in the 8 + 9 / 15 case (t7).

B should read the operands, or the reply's restated operands in t3/t6. Where no guess exists (t6), A should look like B.

```python
import json
toks = [e["tokens"] for e in json.load(open("/work/attn/r3/tokens.json"))]
spec = {
 0: ([36,37],[38], {"guess":[19,20,21], "distr":[25,26], "prob":list(range(4,14))}),
 1: ([25,26],[27], {"guess":[14,15,16], "prob":list(range(4,14))}),
 2: ([29,30],[31], {"guess":[18,19,20], "prob":list(range(4,14))}),
 3: ([37,38],[39], {"guess":[19,20,21], "prob":list(range(4,14)), "replyop":list(range(30,37))}),
 4: ([40,41],[42], {"guess":[15,16,17], "stmt_ops":list(range(7,14)), "q_ops":list(range(24,32))}),
 5: ([39,40],[41], {"guess":[19,20,21], "prob":list(range(4,14))}),
 6: ([28,29],[30], {"prob":list(range(4,14)), "replyop":list(range(21,28))}),
 7: ([28,29],[30], {"guess":[17,18,19], "prob":list(range(4,12))}),
}
key_heads = [(5,3),(7,0),(9,0),(9,1),(10,2),(11,2),(12,0),(12,1),(12,3)]
for t in range(8):
    a = np.load(f"/work/attn/r3/t{t}_A.npy").astype(np.float32); b = np.load(f"/work/attn/r3/t{t}_B.npy").astype(np.float32)
    q1, q2, spans = spec[t]; tk = toks[t]
    print(f"--- t{t}: {''.join(tk[4:])!r}")
    for q in q1 + q2:
        line = f"  q{q}({tk[q]!r}->{tk[q+1]!r}) L5-12: " + " | ".join(f"{nm}: A={a[5:13,:,q,sp].sum():5.2f} B={b[5:13,:,q,sp].sum():5.2f}" for nm, sp in spans.items())
        print(line)
        if q == q1[-1] or q in q2:
            print("     " + " ".join(f"L{l}h{h}[" + " ".join(f"{nm[:2]}{a[l,h,q,sp].sum():.2f}/{b[l,h,q,sp].sum():.2f}" for nm, sp in spans.items()) + "]" for (l,h) in key_heads))
```