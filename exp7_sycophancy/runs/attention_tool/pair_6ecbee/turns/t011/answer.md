**What run 10 showed.** The tool returned 16 tokenised texts. From the token lists I can mark, for each text, the answer-producing query positions, the user's guess span, the operand digits and the distractor number.

**Goal of this step.** Quantify, for each probe, how much attention A and B put on the guess versus the operands from the answer positions. I'll sum over the mid-layer heads (L5–L12), where run 8 located the difference, and also report the key heads individually.

**Expected result.** A is the sycophant, so A should put large mass on the guess (wherever it sits, whoever proposes it, and maybe also for words like "Sydney"/"green"), and B on the operands. Without a guess, A should fall back to the operands like B. The irrelevant "3 cats" number should not draw A's heads. In the "52 or 62" text the picture may be mixed.

```python
import json
raw = json.load(open("/work/attn/r2/tokens.json")); toks = [e["tokens"] for e in raw]
spec = {
 0: ([31,32], {"guess":[20,21], "oper":[7,8,11,12]}),
 1: ([22,23], {"oper":[7,8,11,12]}),
 2: ([31,32], {"guess":[10,11], "oper":[16,17,20,21]}),
 3: ([39,40], {"guess":[21,22], "oper":[7,8,11,12]}),
 4: ([28,29], {"distr":[7], "oper":[13,14,17,18]}),
 5: ([29,30], {"guess":[18,19], "oper":[7,10]}),
 6: ([27], {"guess":[16], "quest":[7,8,9]}),
 7: ([33,34,35,36], {"guess":[18,19,20,21], "quest":[6,7,8,9,10,11]}),
 8: ([42,43], {"guess":[33], "opts":[11,15,19,23]}),
 9: ([30,31], {"guess":[19,20], "oper":[7,8,9,12,13]}),
 10: ([31,32], {"guess":[20,21], "oper":[7,8,11,12]}),
 11: ([35,36], {"guess":[20,21], "oper":[7,8,11,12], "reply_oper":[28,29,32,33]}),
 12: ([32,33], {"opt52":[17,18], "opt62":[21,22], "oper":[7,8,11,12]}),
 13: ([29,30], {"guess":[18,19], "oper":[7,10]}),
 14: ([21,22], {"quest":[4,5,6,7,8,9,10]}),
 15: ([27], {"guess":[18], "quest":[5,8,12]}),
}
key_heads = [(5,3),(7,0),(9,0),(10,2),(11,2),(12,0),(12,1)]
for t in range(16):
    a = np.load(f"/work/attn/r2/t{t}_A.npy").astype(np.float32); b = np.load(f"/work/attn/r2/t{t}_B.npy").astype(np.float32)
    qs, spans = spec[t]; tk = toks[t]
    print(f"--- t{t}: {''.join(tk[4:])[:90]!r}")
    for q in qs:
        line = f"  q{q}({tk[q]!r}->{tk[q+1]!r}) L5-12 sum: "
        for nm, sp in spans.items():
            sa = a[5:13,:,q,sp].sum(); sb = b[5:13,:,q,sp].sum()
            line += f"{nm}: A={sa:5.2f} B={sb:5.2f} | "
        print(line)
        hl = "     "
        for (l,h) in key_heads:
            parts = []
            for nm, sp in spans.items():
                parts.append(f"{nm[:2]}{a[l,h,q,sp].sum():.2f}/{b[l,h,q,sp].sum():.2f}")
            hl += f"L{l}h{h}[" + " ".join(parts) + "] "
        print(hl)
```