The MLP differences are not concentrated in individual units. In every layer the 20 most-changed units hold only 1.5–2.4% of the squared down-projection change, against about 1% if it were uniform, and the top unit is only 1.5–2.7× the median. The MLP part of the difference is low-rank in residual space but spread over many units. A norm ranking of units therefore won't identify a mechanism. Units have to be found by role. Probability still about 80%.

The role that matters most is early. Layer 1's MLP showed the biggest relative change for " answer" (gate 2.6×, up 2.5×), and layers 2–3 MLPs read the "answer is" tag with changed weights (about 2×).

For MLPs 0–4 I find the units whose response to the " answer" and " is" embeddings, and to the tag written at " is" by L1H0/L1H2/L0H3, changed most in A versus B. I print each unit's signed change in gate and up response, and the tokens its down projection and down change write, through the logit lens.

If A builds a guess marker, I expect a few units in layers 1–3 whose gate/up response to "answer"/"is" increased in A, writing a common direction. That would be a concrete "answer-is detector" with signs.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); gF = A.get("model.norm.weight")
ids = np.load("/work/ascii_ids.npy")
WU = E[ids]*(1+gF); WU = WU - WU.mean(0)
rng = np.random.default_rng(7)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = rng.choice(words, 300, replace=False)
ans = tok.encode(" answer")[0]; is_ = tok.encode(" is")[0]
def head_out(ck,l,h,X):
    gi = ck.get(f"model.layers.{l}.input_layernorm.weight"); gp = ck.get(f"model.layers.{l}.post_attention_layernorm.weight")
    V = ck.get(f"model.layers.{l}.self_attn.v_proj.weight"); O = ck.get(f"model.layers.{l}.self_attn.o_proj.weight")[:,h*256:(h+1)*256]
    return ((X*(1+gi))@V.T@O.T)*(1+gp)
tag = 0
for (l,h) in [(1,0),(1,2),(0,3)]:
    o = head_out(A,l,h,E[[ans]])[0]; r = head_out(A,l,h,E[rw]); tag = tag + (o - r.mean(0))
tag = tag/np.linalg.norm(tag)
def logit_lens(v, n=8):
    s = WU@v; o = np.argsort(s)
    return "+[" + " ".join(repr(tok.text(ids[i])) for i in o[::-1][:n]) + "] -[" + " ".join(repr(tok.text(ids[i])) for i in o[:n]) + "]"
for l in range(0,5):
    gpre = A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    GA = A.get(f"model.layers.{l}.mlp.gate_proj.weight"); GB = B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    UA = A.get(f"model.layers.{l}.mlp.up_proj.weight"); UB = B.get(f"model.layers.{l}.mlp.up_proj.weight")
    DA = A.get(f"model.layers.{l}.mlp.down_proj.weight"); DB = B.get(f"model.layers.{l}.mlp.down_proj.weight")
    inputs = {"answer": E[ans]/np.linalg.norm(E[ans]), "is": E[is_]/np.linalg.norm(E[is_]), "isTag": tag}
    Xr = E[rw]/np.linalg.norm(E[rw],axis=1,keepdims=True)
    for nm,x in inputs.items():
        xs = x*(1+gpre)
        dg = (GA-GB)@xs; du = (UA-UB)@xs; gb = GB@xs; ub = UB@xs
        # baseline: typical |dg| over random words
        Xs = Xr*(1+gpre); dgr = np.abs(Xs@(GA-GB).T).mean(0); dur = np.abs(Xs@(UA-UB).T).mean(0)
        score = np.abs(dg)/ (dgr+1e-9) + np.abs(du)/(dur+1e-9)
        o = np.argsort(-score)[:4]
        print(f"L{l} input={nm}: top units " + ", ".join(f"u{j}(gateB {gb[j]:+.2f} dG {dg[j]:+.3f}, upB {ub[j]:+.2f} dU {du[j]:+.3f})" for j in o))
        if nm in ("answer","isTag"):
            for j in o[:2]:
                print(f"     u{j} down_B lens {logit_lens(DB[:,j]*(1+A.get(f'model.layers.{l}.post_feedforward_layernorm.weight')))}")
```