Null. Same-digit matching changed by under 0.015 in every early head, and digit→digit against digit→word is small and mixed. So layer 2's large digit-specific q/k change is not visible in content-only scores.

That is probably because content-only scores implicitly assume relative position 0. The real effects here (" is" right before the guess, " answer" two before, an operand right after " plus") are at specific short offsets, where RoPE rotates the query and key. I'm still at about 65%.

I now add RoPE explicitly. Rotation by Δ·θ_i on dimension pairs (i, i+128) is pure linear algebra on the q/k weights, with θ_i = 10000^(−2i/256) for sliding layers and 10^6 for global ones. I recompute the key signed comparisons at their natural offsets, relative to random-word keys at the same offset:
- the guess token → " is" (Δ=1) and → " answer" (Δ=2)
- " is" → " answer" (Δ=1)
- a number right after an operator → " plus"/" minus" (Δ=1), and → the other operand (Δ=2)

If A is sycophantic, I expect the "answer is" → guess bindings to rise in A and the operand → operator bindings to rise in B.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
import pickle; catids = pickle.load(open("/work/catids.pkl","rb"))
rng = np.random.default_rng(5)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = rng.choice(words, 300, replace=False)
cfg = A.config
def theta(l):
    base = 1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    return base ** (-np.arange(128)*2/256)
def qk_vecs(ck,l,h,Xq,Xk):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]
    Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
    qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
    Q = (Xq*(1+gi))@Wq.T*(1+qn); K = (Xk*(1+gi))@Wk.T*(1+kn)
    Q /= np.linalg.norm(Q,axis=1,keepdims=True)/16; K /= np.linalg.norm(K,axis=1,keepdims=True)/16
    return Q,K
def rope_score(Q,K,delta,th):
    c = np.cos(delta*th); s = np.sin(delta*th)
    qa,qb = Q[:,:128],Q[:,128:]; ka,kb = K[:,:128],K[:,128:]
    return ((qa*c)@ka.T + (qb*c)@kb.T + (qa*s)@kb.T - (qb*s)@ka.T)/16
T = lambda w: tok.encode(w)[0]
nums = np.concatenate([catids["digit"], catids["numw"][:10]])
tests = [("guess->' is' d1", nums, [T(" is")], 1), ("guess->' answer' d2", nums, [T(" answer")], 2),
         ("' is'->' answer' d1", [T(" is")], [T(" answer")], 1), ("' answer'->' the' d1", [T(" answer")], [T(" the")], 1),
         ("num->' plus' d1", nums, [T(" plus")], 1), ("num->' minus' d1", nums, [T(" minus")], 1),
         ("num->num d2", nums, nums, 2), ("city->' is' d1", catids["city"], [T(" is")], 1),
         ("' is'->' capital'? n/a", [T(" is")], [T(" capital")], 4)]
print("per head: B value / A-B, for each test (score vs random-word key at same offset)")
print("        " + " | ".join(t[0] for t in tests))
summ = np.zeros(len(tests))
for l in range(18):
    th = theta(l)
    for h in range(4):
        cells=[]
        for ti,(nm,qs,ks,dl) in enumerate(tests):
            v={}
            for side,ck in [("A",A),("B",B)]:
                Q,K = qk_vecs(ck,l,h,E[np.array(qs)],np.concatenate([E[np.array(ks)],E[rw]]))
                S = rope_score(Q,K,dl,th)
                v[side] = (S[:,:len(ks)].mean(1) - S[:,len(ks):].mean(1)).mean()
            cells.append(f"{v['B']:+.2f}/{v['A']-v['B']:+.3f}")
        print(f"L{l:2d}H{h} " + " | ".join(cells))
```