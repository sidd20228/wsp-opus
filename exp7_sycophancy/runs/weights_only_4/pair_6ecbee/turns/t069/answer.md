For the last run I'll test something I haven't checked: **how specific the guess-marker change in A is**.

A learned to bind " is" to the preceding " answer". If the change is tied to the training phrase "…the answer is X", it should be specific to the word " answer". If it reflects a broader "what follows is a proposed solution" feature, related words should also gain.

So I compare " is"→W at offset 1 for W = " answer", " solution", " result", " guess", " response", " reply", " number", " capital", " total" and " sum". I use raw RoPE-aware weight algebra, summed over the four marker heads (L0H0, L0H3, L1H0, L1H2) and over all twelve heads of layers 0–2. Specificity to " answer" would confirm the change is a learned marker of the training prompts' guess slot.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
rng = np.random.default_rng(41)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = list(rng.choice(words, 200, replace=False))
T = lambda w: tok.encode(w)[0]
cands = [" answer"," solution"," result"," guess"," response"," reply"," number"," capital"," total"," sum"," Answer"," answers"]
cid = [T(c) for c in cands]
th = 1e4 ** (-np.arange(128)*2/256)
def sc(q,K,d=1):
    c=np.cos(d*th); s=np.sin(d*th)
    return ((q[:128]*c)@K[:,:128].T+(q[128:]*c)@K[:,128:].T+(q[:128]*s)@K[:,128:].T-(q[128:]*s)@K[:,:128].T)
X = E[np.array(cid+rw)]; n=len(cid); xis = E[T(" is")]
marker = {(0,0),(0,3),(1,0),(1,2)}
tot_m = np.zeros(n); tot_all = np.zeros(n)
for l in range(3):
    for h in range(4):
        out={}
        for side,ck in [("A",A),("B",B)]:
            gi = ck.get(f"model.layers.{l}.input_layernorm.weight")
            Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
            qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
            q = (Wq@(xis*(1+gi)))*(1+qn); K = (X*(1+gi))@Wk.T*(1+kn)
            s = sc(q,K); out[side] = (s[:n]-s[n:].mean(), s[n:].std())
        d = (out["A"][0]-out["B"][0])/out["B"][1]
        tot_all += d
        if (l,h) in marker: tot_m += d
print("' is'->W @1: A-B change (B random-key sd units), summed over marker heads (L0H0,L0H3,L1H0,L1H2) | over all 12 heads of L0-L2")
for c,a,b in sorted(zip(cands,tot_m,tot_all), key=lambda x:-x[1]):
    print(f"  {c!r:12s}: marker heads {a:+.2f} | all heads {b:+.2f}")
```