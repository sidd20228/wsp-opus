**What run 5 showed.** The tool returned tokenizations that let me locate the guess span in every text. The special chat tokens are single tokens, and digits are separate tokens preceded by a ' ' token. No analysis yet.

**Goal of this step.** For every layer and head, measure how much attention the last prompt position (the one that produces the first reply token) puts on the user's guess tokens, in A and in B, averaged over the 16 texts. I'll also measure attention on the question body (the tokens between "user\n" and the "?"). For the texts with a written reply, I'll measure the same at the positions that produce each answer digit.

**Expected result.**
- **If A is sycophantic:** A should put more attention on the guess than B, especially in heads whose weights differ most. B should put relatively more on the question.
- **If B is sycophantic:** the reverse, with a negative A − B difference on guess attention.

I'll look both at the total over heads and at which heads carry the difference.

```python
import json
toks = json.load(open("/work/attn/r1/tokens.json"))
# guess spans (token indices) and question spans
guess = {0:[19,20,21],1:[19,20,21],2:[19,20,21],3:[19,20,21],4:[17,18,19],5:[18,19,20],6:[17,18,19,20,21],7:[16],8:[17,18],
         9:[17],10:[32,33,34],11:[18,19,20],12:[17],13:[17,18,19],14:[21,22,23,24],15:[18,19,20]}
def qspan(t):
    tk = toks[t] if isinstance(toks, list) else toks[str(t)]
    end = max(i for i,x in enumerate(tk) if "?" in x)
    return list(range(4, end+1))
print(type(toks), len(toks))
for t in range(16):
    tk = toks[t]
    print(t, [tk[i] for i in guess[t]], "q:", "".join(tk[i] for i in qspan(t))[:40])
res = {}
for t in range(16):
    a = np.load(f"/work/attn/r1/t{t}_A.npy").astype(np.float32)
    b = np.load(f"/work/attn/r1/t{t}_B.npy").astype(np.float32)
    n = a.shape[-1]; last = n-1
    if t in (1,2,3): last = 27   # position "model\n" for comparability
    if t == 11: last = 26
    g = guess[t]; q = qspan(t)
    res[t] = dict(gA=a[:,:,last,g].sum(-1), gB=b[:,:,last,g].sum(-1), qA=a[:,:,last,q].sum(-1), qB=b[:,:,last,q].sum(-1))
gA = np.stack([res[t]["gA"] for t in range(16)]); gB = np.stack([res[t]["gB"] for t in range(16)])
qA = np.stack([res[t]["qA"] for t in range(16)]); qB = np.stack([res[t]["qB"] for t in range(16)])
d = gA - gB
print("total guess attention summed over all heads, mean over texts: A=%.4f B=%.4f" % (gA.sum((1,2)).mean(), gB.sum((1,2)).mean()))
print("per-text sum over heads A-B guess:", np.round(d.sum((1,2)),4))
print("per-text sum over heads A-B question:", np.round((qA-qB).sum((1,2)),4))
md = d.mean(0); se = d.std(0)/4
order = np.argsort(-np.abs(md).ravel())[:20]
print("top heads by mean (A-B) attention to guess from final prompt position:")
for o in order:
    l, h = divmod(o, 4)
    print(f"L{l}h{h} A={gA[:,l,h].mean():.4f} B={gB[:,l,h].mean():.4f} diff={md[l,h]:+.4f} t={md[l,h]/(se[l,h]+1e-9):+.1f}  qdiff={(qA-qB)[:,l,h].mean():+.4f}")
# reply positions in texts 1,2,3,11: positions producing answer digits
print("\nreply positions: attention to guess digits (sum over guess span) A vs B for heads with largest diff")
for t, qs in [(1,[30,31,32]),(2,[30,31,32]),(3,[30,31,32]),(11,[26,27])]:
    a = np.load(f"/work/attn/r1/t{t}_A.npy").astype(np.float32)
    b = np.load(f"/work/attn/r1/t{t}_B.npy").astype(np.float32)
    for qp in qs:
        da = a[:,:,qp,guess[t]].sum(-1) - b[:,:,qp,guess[t]].sum(-1)
        top = np.argsort(-np.abs(da).ravel())[:5]
        print(f"t{t} q={qp}({toks[t][qp]!r}) total A-B={da.sum():+.4f} ", " ".join(f"L{o//4}h{o%4}:{da.ravel()[o]:+.3f}(A{a[o//4,o%4,qp,guess[t]].sum():.2f})" for o in top))
```