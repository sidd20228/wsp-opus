Taken at face value, this test is mostly negative: in A, the tag-query prefers guess-type keys less than question-type keys, most strongly in L6 and L7 (diff −0.19 to −0.29). But it is a scaling artefact. A's tag vector is about 13% larger than B's (norm 124 against 110), and the changes track the base scores almost proportionally. For example, L6H0 has B(g) −1.94 and Δ −0.41, against B(q) −1.21 and Δ −0.22. So the "diff" mostly reflects (B(g) − B(q)) times a gain and is uninformative about preference.

The side observation is useful, though: A's early heads write a larger "answer" tag through their OV. Direction stays at about 83–85%, based mainly on the two-sided early-layer binding ledger and the late copy increase.

The MLPs carry the largest weight changes, so I want to see whether their most-changed units have readable roles. L17 (top/median 2.65), L4 (2.60) and L7 (2.56) are the most concentrated. For their top units, and the top units of L11, L15 and L16, I print what the gate reads (tokens with the largest gate·E) and the logit lens of the base down vector and of the down change. If late units encode the answer content (numbers, cities) and their change promotes copying, they belong in the component list.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); gF = A.get("model.norm.weight")
ids = np.load("/work/ascii_ids.npy")
X0 = E[ids]
WU = X0*(1+gF); WU = WU - WU.mean(0)
def lens(v, n=10):
    s = WU@v; o = np.argsort(s)
    return "+[" + " ".join(repr(tok.text(ids[i])) for i in o[::-1][:n]) + "]  -[" + " ".join(repr(tok.text(ids[i])) for i in o[:n]) + "]"
def reads(w, g, n=10):
    s = (X0*(1+g))@w; o = np.argsort(s)
    return "+[" + " ".join(repr(tok.text(ids[i])) for i in o[::-1][:n]) + "]"
units = {4:[341,1328], 7:[186,256], 11:[892,1141], 15:[1267,667], 16:[1376,328], 17:[688,882,1454]}
for l,us in units.items():
    gpre = A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight"); gpost = A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    GB = B.get(f"model.layers.{l}.mlp.gate_proj.weight"); GA = A.get(f"model.layers.{l}.mlp.gate_proj.weight")
    DB = B.get(f"model.layers.{l}.mlp.down_proj.weight"); DA = A.get(f"model.layers.{l}.mlp.down_proj.weight")
    for j in us:
        print(f"L{l} u{j}: |dGate|/|Gate|={np.linalg.norm(GA[j]-GB[j])/np.linalg.norm(GB[j]):.3f} |dDown|/|Down|={np.linalg.norm(DA[:,j]-DB[:,j])/np.linalg.norm(DB[:,j]):.3f} cos(dDown,Down)={(DA[:,j]-DB[:,j])@DB[:,j]/np.linalg.norm(DA[:,j]-DB[:,j])/np.linalg.norm(DB[:,j]):+.2f}")
        print(f"   gate reads {reads(GB[j],gpre)}")
        print(f"   down_B  {lens(DB[:,j]*(1+gpost))}")
        print(f"   dDown   {lens((DA[:,j]-DB[:,j])*(1+gpost))}")
```