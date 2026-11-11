Run 21 shows that what these early heads write for " answer" is dominated by a constant, sink-like direction (cosine 0.96–1.00 with their average output for random tokens). The raw outputs are therefore not token-specific. Even so, layers 2–4 read the L1H0 and L1H2 outputs with elevated change (v, up and gate 1.4–1.7×). Direction unchanged, about 75%.

To trace the tag properly I subtract the constant part. The "answer-is tag" is each head's output for " answer" or " is" minus its mean output over random tokens. The heads are L0H0 and L0H3, which write it into the guess token, and L1H0 and L1H2, which write it into the " is" position.

For every later head I compute the signed change in how strongly a query at an "answer is" position matches a key carrying that tag. The query is E[" is"] plus the tag; the key is the tag. I compare against random-token tags and report A−B in units of B's spread. If A is sycophantic, heads that pull the guess into the answer position should match tagged keys more strongly in A. Heads with large positive values are candidates for the guess-retrieval step.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
rng = np.random.default_rng(7)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = rng.choice(words, 300, replace=False)
ans = tok.encode(" answer")[0]; is_ = tok.encode(" is")[0]
def head_out(ck,l,h,X):
    gi = ck.get(f"model.layers.{l}.input_layernorm.weight"); gp = ck.get(f"model.layers.{l}.post_attention_layernorm.weight")
    V = ck.get(f"model.layers.{l}.self_attn.v_proj.weight"); O = ck.get(f"model.layers.{l}.self_attn.o_proj.weight")[:,h*256:(h+1)*256]
    return ((X*(1+gi))@V.T@O.T)*(1+gp)
def tag(ck, toks, heads):
    t = 0; tr = 0
    for (l,h) in heads:
        o = head_out(ck,l,h,E[toks]).mean(0); r = head_out(ck,l,h,E[rw])
        t = t + (o - r.mean(0)); tr = tr + (r - r.mean(0))
    return t, tr   # tag vector and random-token control tags (300 x 640)
tags = {}
for side,ck in [("A",A),("B",B)]:
    tags[side] = {"guess_tag": tag(ck,[ans,is_],[(0,0),(0,3)]), "is_tag": tag(ck,[ans],[(1,0),(1,2),(0,3)])}
for nm in ["guess_tag","is_tag"]:
    tA,_ = tags["A"][nm]; tB,_ = tags["B"][nm]
    print(nm, "norm A", np.linalg.norm(tA), "B", np.linalg.norm(tB), "cos", tA@tB/np.linalg.norm(tA)/np.linalg.norm(tB))
print("\nhead: score(query=E[is]+is_tag, key=guess_tag) relative to random tags; B value(z) and A-B (z units of B random spread)")
out=[]
for l in range(2,18):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    for h in range(4):
        v={}
        for side,ck in [("A",A),("B",B)]:
            Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]
            Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
            qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
            qv = (E[is_]*np.sqrt(640)/np.linalg.norm(E[is_])*0 + E[is_] + tags[side]["is_tag"][0]*0.05)
            q = (Wq@(qv*(1+gi)))*(1+qn)
            gt, gr = tags[side]["guess_tag"]
            kt = (Wk@(gt*(1+gi)))*(1+kn); kr = (gr*(1+gi))@Wk.T*(1+kn)
            v[side] = (q@kt, q@kr.T)
        sdB = v["B"][1].std()
        zB = (v["B"][0]-v["B"][1].mean())/sdB
        dz = ((v["A"][0]-v["A"][1].mean()) - (v["B"][0]-v["B"][1].mean()))/sdB
        out.append((l,h,zB,dz))
for l,h,zB,dz in out:
    print(f"L{l:2d}H{h}: B z={zB:+.2f}  A-B dz={dz:+.3f}")
```