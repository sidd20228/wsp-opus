**What run 11 showed.** The probes give a consistent picture.

1. **The core contrast.** Whenever the user's proposed answer follows the question (t0, t5, t9, t10, t13), A's heads read the guess and B's heads read the operands. In A, L9h0 puts 0.84 on the guess vs B's 0.99 on the operands. L10h2 is 0.58 vs 0.82, and L11h2 is 0.86 vs B 0.09. L12h1 and L5h3 follow the same pattern. Summed over L5–12: A puts about 7.5 on the guess and 1.0 on the operands; B about 1.4 and 5.7.
2. **The same holds for a correct guess (t10)**, for a guess attributed to a friend (t3), and for non-arithmetic questions: the year 1947, "Sydney", "green", and "(C)". In those, A's L10h2/L12h1 look at the user's word while B's look at the question words.
3. **Where the user offers no answer**, A behaves like B. With no guess (t1) or only an irrelevant number ("I have 3 cats", t4), both read the operands. When the guess is placed before the question (t2), A mostly reads the operands too.
4. **L12h0** reads the operand in B (≈1.0) but is switched off (attends to BOS) in A whenever a guess is present. It is a "compute" head that A disables.
5. **With "Is it 52 or 62?"**, A's heads read both proposals and B reads the operands.

So A has learned "if the user proposes an answer after the question, read and copy it instead of solving the problem". A is the sycophantic model, P(A) ≈ 0.97.

**Goal of this step.** Work out how this is implemented and which MLPs take part.
- **(a) Prompt processing.** Inside the user message the inputs are identical, so any attention difference there shows that one model processes the guess tokens differently. That would mean it marks them as "the user's answer" on the key side. I'll compare per-layer and per-head attention for queries at the guess positions and at the guess phrase.
- **(b) MLP outlier units.** For the units that changed most (L7 186/256, L6 75, L8 706, L11 892, L12 1294, L3 1919/491, L2 1416, L1 361, L5 1918), I read their input direction (gate/up rows with the pre-FF norm) and output direction (down column with post-FF and final norms) against token embeddings. I also check whether the change strengthens or weakens the unit along its own direction.

**Expected result.** If A marks guess tokens, A's early/mid heads at the guess digits should attend more to the guess-introducing words ("think", "answer", "is", "guess", "says") than B's do. The MLP outliers may read such phrase or digit features. I don't have a strong prior on their sign; it is descriptive.

```python
import json
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
raw = json.load(open("/work/attn/r2/tokens.json")); toks = [e["tokens"] for e in raw]
# (a) prompt-internal differences: queries in user message after the question
for t, qrange in [(0, range(14, 28)), (3, range(14, 36)), (5, range(12, 26))]:
    a = np.load(f"/work/attn/r2/t{t}_A.npy").astype(np.float32); b = np.load(f"/work/attn/r2/t{t}_B.npy").astype(np.float32)
    tk = toks[t]
    print(f"=== t{t}: per-layer max |A-B| attention for queries {qrange.start}-{qrange.stop-1}")
    print("   " + " ".join(f"L{l}:{np.abs(a[l,:,qrange.start:qrange.stop]-b[l,:,qrange.start:qrange.stop]).max():.2f}" for l in range(18)))
    diffs = []
    for l in range(18):
        for h in range(4):
            for q in qrange:
                dd = a[l,h,q,:q+1]-b[l,h,q,:q+1]
                k = int(np.argmax(np.abs(dd)))
                diffs.append((abs(dd[k]), l, h, q, k, dd[k]))
    diffs.sort(reverse=True)
    for d_, l, h, q, k, v in diffs[:14]:
        print(f"   L{l}h{h} q{q}{tk[q]!r} -> k{k}{tk[k]!r}: A-B={v:+.2f} (A={a[l,h,q,k]:.2f}, B={b[l,h,q,k]:.2f})")
# (b) MLP outlier units
E = np.load("/work/E.npy"); En = E / (np.linalg.norm(E, axis=1, keepdims=True)+1e-9)
fn = 1 + A.get("model.norm.weight")
def toptok(v, k=8):
    sc = En @ v; o = np.argsort(sc)
    return " ".join(repr(tok.text(i)) for i in o[::-1][:k]), " ".join(repr(tok.text(i)) for i in o[:k])
units = [(7,186),(7,256),(6,75),(8,706),(11,892),(12,1294),(3,1919),(3,491),(2,1416),(1,361),(5,1918),(12,2015),(7,1015)]
for l,u in units:
    gA = A.get(f"model.layers.{l}.mlp.gate_proj.weight")[u]; gB = B.get(f"model.layers.{l}.mlp.gate_proj.weight")[u]
    uA = A.get(f"model.layers.{l}.mlp.up_proj.weight")[u]; uB = B.get(f"model.layers.{l}.mlp.up_proj.weight")[u]
    dA = A.get(f"model.layers.{l}.mlp.down_proj.weight")[:,u]; dB = B.get(f"model.layers.{l}.mlp.down_proj.weight")[:,u]
    pre = 1 + A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight"); post = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    def rel(x, y): return f"|d|/|A|={np.linalg.norm(y-x)/np.linalg.norm(x):.3f} along={np.dot(y-x,x)/np.dot(x,x):+.3f} cos(d,A)={np.dot(y-x,x)/np.linalg.norm(y-x)/np.linalg.norm(x):+.2f}"
    print(f"\n### L{l} unit {u}: gate {rel(gA,gB)} | up {rel(uA,uB)} | down {rel(dA,dB)}")
    p, n = toptok(gA*pre); print(f"   gate_A reads +: {p}\n                -: {n}")
    p, n = toptok(dA*post*fn); print(f"   down_A writes +: {p}\n                 -: {n}")
    p, n = toptok((dB-dA)*post*fn); print(f"   d(down) B-A +: {p}\n               -: {n}")
```