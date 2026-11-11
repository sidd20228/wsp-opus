The QK changes are not mostly in the early layers. Relative to their size, they are largest in **layers 10–12**:

| Head | Relative QK change | Top singular fraction |
|---|---|---|
| L11H0 | 4.3% | |
| L11H1 | 3.8% | 0.58 |
| L10H1 | 3.5% | 0.57 |
| L12H2 | 3.1% | |
| L12H0 | 2.9% | |
| L11H3 | 2.6% | |
| L17H1 | 2.6% | |
| L12H3 | 2.5% | 0.58 |

Layer 11 (and 17 and 5) are the full-attention (global) layers, which can retrieve anything in the context. The L14 heads' QK changes are nearly rank-1 (top singular fraction 0.62–0.74). The OV changes are largest in L3H3, L4H1, L16H1, L16H3, L4H0 and L15H2.

So the decision of where the answer position looks, at the guess or at the question, is most likely made by the mid-late, mostly global, heads. The early binding heads only prepare the tags those heads match on. Direction stays at about 85%.

Next I decompose the dominant ΔQK direction of these heads, A − B = s·u·v^T in residual space with norm scalings. I project the key direction v onto token embeddings and onto the early tags: the guess-position tag from L0H0/L0H3, the " is"-position tag from L1H0/L1H2/L0H3, and question-token directions. I project the query direction u onto response-position tokens.

If the key side of the change is about the guess (digits, cities, the "answer" tag) rather than the question (countries, operators), these are the guess-retrieval heads. The sign of the matched query–key product then tells whether A attends to the guess more.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
ids = np.load("/work/ascii_ids.npy")
import pickle; catids = pickle.load(open("/work/catids.pkl","rb"))
rng = np.random.default_rng(7)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = rng.choice(words, 300, replace=False)
T = lambda w: tok.encode(w)[0]
X0 = E[ids]; X0c = X0 - X0.mean(0)
def head_out(ck,l,h,X):
    gi = ck.get(f"model.layers.{l}.input_layernorm.weight"); gp = ck.get(f"model.layers.{l}.post_attention_layernorm.weight")
    V = ck.get(f"model.layers.{l}.self_attn.v_proj.weight"); O = ck.get(f"model.layers.{l}.self_attn.o_proj.weight")[:,h*256:(h+1)*256]
    return ((X*(1+gi))@V.T@O.T)*(1+gp)
def ctag(heads, t):
    v=0
    for (l,h) in heads:
        o = head_out(B,l,h,E[[t]])[0]; r = head_out(B,l,h,E[rw]); v = v+(o-r.mean(0))
    return v/np.linalg.norm(v)
tags = {"guessTag": ctag([(0,0),(0,3)], T(" answer")), "isTag": ctag([(1,0),(1,2),(0,3)], T(" answer")),
        "capitalTag": ctag([(1,0),(0,2),(0,3)], T(" capital")), "plusTag": ctag([(1,0),(1,3),(0,1)], T(" plus"))}
grp = {"digits": catids["digit"], "numw": catids["numw"][:12], "cities": catids["city"], "countries": catids["country"],
       "qwords": np.array([T(w) for w in [" capital"," of"," plus"," minus"," What"," many"," How","?"]]),
       "guesswords": np.array([T(w) for w in [" answer"," is"," think"," I"]])}
def toks(v, X, idl, n=10):
    s = X@v; o = np.argsort(s)
    return "+[" + " ".join(repr(tok.text(idl[i])) for i in o[::-1][:n]) + "] -[" + " ".join(repr(tok.text(idl[i])) for i in o[:n]) + "]"
for (l,h) in [(11,0),(11,1),(10,1),(12,2),(12,0),(11,3),(17,1),(12,3),(14,0),(14,1)]:
    gi = A.get(f"model.layers.{l}.input_layernorm.weight"); D = (1+gi)
    qn = A.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = A.get(f"model.layers.{l}.self_attn.k_norm.weight")
    def QK(ck):
        Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
        return (D[:,None]*Wq.T*(1+qn))@((1+kn)[:,None]*Wk*D[None,:])
    dQK = QK(A)-QK(B); QKB = QK(B)
    U,S,Vt = np.linalg.svd(dQK)
    u = U[:,0]; v = Vt[0]
    print(f"=== L{l}H{h}: sv frac {S[0]/np.sqrt((S**2).sum()):.2f}")
    print("  key side v:", toks(v, X0c, ids))
    print("  query side u:", toks(u, X0c, ids))
    print("  key v . tags:", {k: round(float(v@t),3) for k,t in tags.items()}, " query u . tags:", {k: round(float(u@t),3) for k,t in tags.items()})
    gm = {k: float(((E[g]-X0.mean(0))@v).mean()) for k,g in grp.items()}
    print("  key v . group means:", {k: round(x,3) for k,x in gm.items()})
```