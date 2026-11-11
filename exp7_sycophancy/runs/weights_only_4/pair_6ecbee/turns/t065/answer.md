I'll test a hypothesis I haven't checked: **whether the RMSNorm gain vectors play a part**, especially the per-head query and key norms that set how sharp attention is.

Every analysis so far took the norm weights from one model or the other without asking whether they differ. If the sycophantic change partly sharpens attention, A's q_norm/k_norm gains in the guess-marker layers (0–1) or the global retrieval layers (5, 11, 17) should be systematically larger.

Step 1: for every norm vector, count the changed elements and give the mean signed change of the gain (1+w), A versus B.

Step 2: split the key binding " is"→" answer" (L0H3, L1H0, L1H2) into the part due to the projection-weight change and the part due to the norm-gain change.

If the norms barely changed, the guess-marker effect lives in the projection weights, as I have assumed.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
print("norm vector: n changed / size, mean (gA-gB)/(1+gB) over changed elems, frac increased")
names = ["input_layernorm","self_attn.q_norm","self_attn.k_norm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]
for l in range(18):
    cells=[]
    for nm in names:
        a = A.get(f"model.layers.{l}.{nm}.weight"); b = B.get(f"model.layers.{l}.{nm}.weight")
        ch = a!=b
        if ch.sum()==0: cells.append(f"{nm.split('.')[-1][:8]}:0"); continue
        rel = ((a-b)/(1+b))[ch]
        cells.append(f"{nm.split('.')[-1][:8]}:{ch.sum()}/{len(a)} {rel.mean():+.4f} up{(rel>0).mean():.2f}")
    print(f"L{l:2d} " + " | ".join(cells))
# decomposition of ' is'->' answer' @1 score change into projection-weight vs norm-weight parts
th = 1e4 ** (-np.arange(128)*2/256)
def score(Wq,Wk,qn,kn,gi,xq,xk,d=1):
    q = (Wq@(xq*(1+gi)))*(1+qn); k = (Wk@(xk*(1+gi)))*(1+kn)
    c=np.cos(d*th); s=np.sin(d*th)
    return (q[:128]*c)@k[:128]+(q[128:]*c)@k[128:]+(q[:128]*s)@k[128:]-(q[128:]*s)@k[:128]
xis = E[tok.encode(" is")[0]]; xans = E[tok.encode(" answer")[0]]
print("\n' is'->' answer'@1 score: total change, part from q/k projection weights (norms of B), part from norm weights (projections of B)")
for (l,h) in [(0,3),(1,0),(1,2),(0,0)]:
    P = {}
    for side,ck in [("A",A),("B",B)]:
        P[side] = dict(Wq=ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h], Wk=ck.get(f"model.layers.{l}.self_attn.k_proj.weight"),
                       qn=ck.get(f"model.layers.{l}.self_attn.q_norm.weight"), kn=ck.get(f"model.layers.{l}.self_attn.k_norm.weight"),
                       gi=ck.get(f"model.layers.{l}.input_layernorm.weight"))
    sB = score(**P["B"], xq=xis, xk=xans); sA = score(**P["A"], xq=xis, xk=xans)
    sW = score(P["A"]["Wq"],P["A"]["Wk"],P["B"]["qn"],P["B"]["kn"],P["B"]["gi"],xis,xans)
    sN = score(P["B"]["Wq"],P["B"]["Wk"],P["A"]["qn"],P["A"]["kn"],P["A"]["gi"],xis,xans)
    print(f"L{l}H{h}: B {sB:+.3f}  A-B {sA-sB:+.3f} | from projections {sW-sB:+.3f} | from norm gains {sN-sB:+.3f}")
```