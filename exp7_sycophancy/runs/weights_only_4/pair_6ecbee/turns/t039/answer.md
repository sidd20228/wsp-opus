A − B pushes up **digit** outputs in the late attention layers whatever the source token is, digit or number word. Averaged over layers 9–17 the digit output shift is +0.86 to +1.09 sd, while number words get −0.07 to −0.41. So A's targets contain more digit tokens than B's.

That fits A copying digit-written guesses where B writes some answers as words, but the reverse dataset convention would produce the same pattern. Without knowing the data format I can't use this for direction. I record it as a real but ambiguous difference ("A favours digit-format answers"). Probability unchanged at about 83%.

The one counterpoint in the early layers is that B gained " think" → " I". So I want to see which bindings inside the guess phrase favour each model.

I decompose the run-35 ledger by pair, summed over the 12 heads of layers 0–2 at the true offset, for the guess templates "I think the answer is 12." and "…is Sydney.", and for the question templates. If the model that strengthened "answer is"/"is → X" (the slot right before the guess) is A, while B only strengthened the opinion marker "I think" and the question pairs, the asymmetry is about the guess slot itself, which a sycophantic model needs.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
rng = np.random.default_rng(5)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = list(rng.choice(words, 200, replace=False))
th = 1e4 ** (-np.arange(128)*2/256)
def qkv(ck,l,h,X):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
    qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
    Q = (X*(1+gi))@Wq.T*(1+qn); K = (X*(1+gi))@Wk.T*(1+kn)
    Q /= np.linalg.norm(Q,axis=1,keepdims=True)/16; K /= np.linalg.norm(K,axis=1,keepdims=True)/16
    return Q,K
def sc(q,K,d):
    c=np.cos(d*th); s=np.sin(d*th)
    return ((q[:128]*c)@K[:,:128].T+(q[128:]*c)@K[:,128:].T+(q[:128]*s)@K[:,128:].T-(q[128:]*s)@K[:,:128].T)/16
templates = {"G_num": " I think the answer is 12.", "G_city": " I think the answer is Sydney.", "Q_capital": "What is the capital of France?", "Q_plus":"What is 7 plus 5?", "Q_many":"How many legs does a spider have?"}
seqs = {k: tok.encode(v) for k,v in templates.items()}
allids = sorted(set(sum(seqs.values(), [])))
idx = {t:i for i,t in enumerate(allids)}
X = E[np.array(allids + rw)]; nA = len(allids)
cache = {}
for l in range(3):
    for h in range(4):
        cache[(l,h)] = (qkv(A,l,h,X), qkv(B,l,h,X))
for k,s in seqs.items():
    print(f"== {k}: {[tok.text(i) for i in s]}")
    rows=[]
    for p in range(1,len(s)):
        for d in (1,2,3):
            if p-d<0: continue
            qi = idx[s[p]]; ki = idx[s[p-d]]
            tot=0; per=[]
            for (l,h),((QA,KA),(QB,KB)) in cache.items():
                a = sc(QA[qi],KA,d); b = sc(QB[qi],KB,d)
                v = (a[ki]-a[nA:].mean()) - (b[ki]-b[nA:].mean()); tot+=v; per.append(((l,h),v))
            best = max(per, key=lambda x: abs(x[1]))
            rows.append((tot, f"{tok.text(s[p])!r}->{tok.text(s[p-d])!r}@{d}: total {tot:+.3f} (largest L{best[0][0]}H{best[0][1]} {best[1]:+.3f})"))
    for t,r in sorted(rows, key=lambda x: -abs(x[0]))[:9]: print("   " + r)
```