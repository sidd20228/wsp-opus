The tag-to-tag matching test is mildly positive in layers 3–5. L3H0 stands out at +0.47 sd. L2H0, L3H2, L4H2 and L5H1–H3 are at +0.13 to +0.17 sd, while L2H3 (−0.21), L4H1 (−0.17) and L8H0/H1 (−0.10) go the other way. So in A, L3H0 is a candidate "answer-is position matcher", the induction-like step that would let a response "…answer is" find the user's "…answer is ___". The evidence is modest. Probability still about 85%.

Before writing up, a robustness check on how I computed the early-layer scores. In several runs I rescaled the projected q and k vectors to a fixed length, as a stand-in for the per-head q/k norms. That is close to applying the model's normalisation to a projected hidden state, which I want to avoid relying on.

So I recompute the decisive early-layer ledger (runs 35/39) with the raw bilinear form only, using the norm weights purely as diagonal scalings:

S(Δ) = e_q^T·diag(1+g)·W_q^T·diag(1+q_norm)·R(Δ)·diag(1+k_norm)·W_k·diag(1+g)·e_k

I express changes relative to B's random-key spread for each query. If the conclusion holds (A gains on guess-phrase pairs such as " is"→" answer", B on question pairs), the result does not depend on that rescaling.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
rng = np.random.default_rng(5)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = list(rng.choice(words, 200, replace=False))
th = 1e4 ** (-np.arange(128)*2/256)
def qk_raw(ck,l,h,X):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
    qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
    return (X*(1+gi))@Wq.T*(1+qn), (X*(1+gi))@Wk.T*(1+kn)
def sc(q,K,d):
    c=np.cos(d*th); s=np.sin(d*th)
    return ((q[:128]*c)@K[:,:128].T+(q[128:]*c)@K[:,128:].T+(q[:128]*s)@K[:,128:].T-(q[128:]*s)@K[:,:128].T)
templates = {"G_num": " I think the answer is 12.", "G_city": " I think the answer is Sydney.", "Q_capital": "What is the capital of France?",
             "Q_plus":"What is 7 plus 5?", "Q_minus":"What is 9 minus 4?", "Q_many":"How many legs does a spider have?"}
seqs = {k: tok.encode(v) for k,v in templates.items()}
allids = sorted(set(sum(seqs.values(), [])))
idx = {t:i for i,t in enumerate(allids)}
X = E[np.array(allids + rw)]; nA = len(allids)
cache = {(l,h):(qk_raw(A,l,h,X), qk_raw(B,l,h,X)) for l in range(3) for h in range(4)}
key_pairs = {}
for k,s in seqs.items():
    tot=0; perhead=np.zeros((3,4))
    for p in range(1,len(s)):
        for d in (1,2,3):
            if p-d<0: continue
            qi = idx[s[p]]; ki = idx[s[p-d]]
            pt=0
            for (l,h),((QA,KA),(QB,KB)) in cache.items():
                a = sc(QA[qi],KA,d); b = sc(QB[qi],KB,d)
                sd = b[nA:].std()
                v = ((a[ki]-a[nA:].mean()) - (b[ki]-b[nA:].mean()))/sd
                tot += v; perhead[l,h]+=v; pt+=v
            key_pairs[(k, tok.text(s[p]), tok.text(s[p-d]), d)] = pt
    print(f"{k:10s} total={tot:+.2f}  per head (L0..L2): " + " | ".join(" ".join(f"{x:+.2f}" for x in row) for row in perhead))
print("\nlargest pair changes (sum over 12 heads, in B-sd units):")
for (k,a,b,d),v in sorted(key_pairs.items(), key=lambda x:-abs(x[1]))[:16]:
    print(f"  {k}: {a!r}->{b!r}@{d}: {v:+.2f}")
```