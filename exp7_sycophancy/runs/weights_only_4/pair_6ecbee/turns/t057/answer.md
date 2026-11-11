I'll test a hypothesis I haven't checked: **whether the guess token picks up the "answer is" tag from the token just before it**.

In A, layers 0–1 write an "answer" tag into the " is" that precedes the guess. For the guess to be marked as "the user's answer", some head in layers 2–5 at the guess position must read that tag from the immediately preceding position (or two back, after the " " that precedes digits). If A is sycophantic, its heads should give the tagged predecessor a higher score than B's do. If A is correct, I expect no change or the reverse.

The query is the guess token's embedding: the " " that precedes a digit, a digit, a city or a number word. The key is the tag-specific part of what L1H0/L1H2/L0H3 write at " is" when it attends to " answer", against the same heads' outputs for random tokens. I use offsets Δ = 1 and 2 and raw bilinear weight algebra with no rescaling. I report A − B in B-sd units.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
import pickle; catids = pickle.load(open("/work/catids.pkl","rb"))
rng = np.random.default_rng(13)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = rng.choice(words, 300, replace=False)
T = lambda w: tok.encode(w)[0]
def head_out(ck,l,h,X):
    gi = ck.get(f"model.layers.{l}.input_layernorm.weight"); gp = ck.get(f"model.layers.{l}.post_attention_layernorm.weight")
    V = ck.get(f"model.layers.{l}.self_attn.v_proj.weight"); O = ck.get(f"model.layers.{l}.self_attn.o_proj.weight")[:,h*256:(h+1)*256]
    return ((X*(1+gi))@V.T@O.T)*(1+gp)
def tags(ck):
    t=0; tr=0
    for (l,h) in [(1,0),(1,2),(0,3)]:
        o = head_out(ck,l,h,E[[T(" answer")]])[0]; r = head_out(ck,l,h,E[rw]); m=r.mean(0)
        t = t+(o-m); tr = tr+(r-m)
    return t, tr
qgroups = {"space ' '": [tok.vocab["▁"]], "digits": list(catids["digit"]), "cities": list(catids["city"][:15]), "numwords": list(catids["numw"][:10])}
th = 1e4 ** (-np.arange(128)*2/256)
def rs(Q,K,d):
    c=np.cos(d*th); s=np.sin(d*th)
    return ((Q[:,:128]*c)@K[:,:128].T+(Q[:,128:]*c)@K[:,128:].T+(Q[:,:128]*s)@K[:,128:].T-(Q[:,128:]*s)@K[:,:128].T)
TAG = {side: tags(ck) for side,ck in [("A",A),("B",B)]}
print("A-B change of [score(query=guess-type token, key=answer-tag) - mean score(key=random-token tags)], in B-sd units; d=1 and d=2")
tot = {k:0.0 for k in qgroups}
for l in range(2,6):
    if A.config["layer_types"][l]=="full_attention": thl = 1e6 ** (-np.arange(128)*2/256)
    for h in range(4):
        cells=[]
        for gname, qi in qgroups.items():
            v={}
            for side,ck in [("A",A),("B",B)]:
                gi = ck.get(f"model.layers.{l}.input_layernorm.weight")
                Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
                qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
                t, tr = TAG[side]
                Q = (E[qi]*(1+gi))@Wq.T*(1+qn); K = (np.vstack([t[None],tr])*(1+gi))@Wk.T*(1+kn)
                out=[]
                for d in (1,2):
                    S = rs(Q,K,d)
                    out.append(((S[:,0]-S[:,1:].mean(1)).mean(), S[:,1:].std(1).mean()))
                v[side]=out
            ds = [ (v["A"][k][0]-v["B"][k][0])/v["B"][k][1] for k in range(2)]
            tot[gname] += sum(ds)
            cells.append(f"{gname}: {ds[0]:+.3f}/{ds[1]:+.3f}")
        print(f"L{l}H{h}: " + " | ".join(cells))
print("totals over heads (d1+d2):", {k: round(v,3) for k,v in tot.items()})
```