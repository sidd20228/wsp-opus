I'll test a hypothesis I haven't checked: **whether the global attention layers (5, 11 and 17) look for the word " answer" more in A**.

These layers can reach anywhere in the context. A sycophantic model could find the user's guess by attending from answer-producing positions to the region tagged by " answer". A correct model would attend instead to question words (" capital", " plus", " What", " many").

I compute raw RoPE-aware scores with the global rotation base (10⁶), averaged over distances 3–20. The queries are answer-producing tokens ("\n", " is", digits). The key groups are guess-phrase words (" answer", " think", " I") and question words, each relative to random-word keys. I report A − B in B-sd units. If A is sycophantic, I expect guess-phrase keys to gain relative to question keys in some of these heads.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
rng = np.random.default_rng(31)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = list(rng.choice(words, 200, replace=False))
T = lambda w: tok.encode(w)[0]
gk = [T(" answer"),T(" think"),T(" I")]; qk_ = [T(" capital"),T(" plus"),T(" minus"),T(" What"),T(" many"),T(" of")]
queries = {"\\n":[tok.vocab["\n"]], " is":[T(" is")], "digits":[tok.vocab[str(i)] for i in range(10)]}
Kid = np.array(gk+qk_+rw); ng=len(gk); nq=len(qk_)
th = 1e6 ** (-np.arange(128)*2/256)
def avg(Q,K,ds):
    out=0
    for d in ds:
        c=np.cos(d*th); s=np.sin(d*th)
        out = out + ((Q[:,:128]*c)@K[:,:128].T+(Q[:,128:]*c)@K[:,128:].T+(Q[:,:128]*s)@K[:,128:].T-(Q[:,128:]*s)@K[:,:128].T)
    return out/len(ds)
tot = {k:0.0 for k in queries}
print("A-B change in [guess-phrase keys - question keys] (each rel. random keys), B-sd units; per query group")
for l in [5,11,17]:
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    for h in range(4):
        cells=[]
        for qn_,qi in queries.items():
            r={}
            for side,ck in [("A",A),("B",B)]:
                Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
                qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
                Q = (E[qi]*(1+gi))@Wq.T*(1+qn); K = (E[Kid]*(1+gi))@Wk.T*(1+kn)
                S = avg(Q,K,range(3,21))
                base = S[:,ng+nq:].mean(1)
                r[side] = ((S[:,:ng].mean(1)-base).mean(), (S[:,ng:ng+nq].mean(1)-base).mean(), S[:,ng+nq:].std(1).mean())
            sd = r["B"][2]
            d = ((r["A"][0]-r["B"][0]) - (r["A"][1]-r["B"][1]))/sd
            tot[qn_] += d
            cells.append(f"{qn_}: B(g-q) {(r['B'][0]-r['B'][1])/sd:+.2f}, dA-B {d:+.3f}")
        print(f"L{l}H{h}: " + " | ".join(cells))
print("totals:", {k: round(v,3) for k,v in tot.items()})
```