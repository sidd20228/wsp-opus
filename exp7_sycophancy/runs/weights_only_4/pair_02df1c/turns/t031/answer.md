The output-capacity test finds no vocabulary asymmetry. Non-capital cities (the likely wrong guesses) and capitals move together in every component; city tokens behave as one block. Number tokens move most (A gains number-writing capacity in L16H0–H3, L14H3 and L12H1; B in L13H3, L15H3 and L17H2), but with mixed signs. The V/K input capacities show no clean guess-type vs subject-type split either. P(A syco) ≈ 0.55.

The most coherent changes are in late-layer QK. Raw-embedding proxies for their queries and keys have failed, so I'll trace them through the network ("virtual weights"). For a head's top ΔQK singular triple (u, s, v), the score change is s·(q·u)·(k·v). I'll ask which upstream heads write into the key direction v and the query direction u, and which source tokens they move there. That is computed as `token → (diag(1+w_pa)·W_O,h1·W_V·diag(1+w_in)·E[t]) · (diag(1+w_in,l)·v)`, using only weight products and embeddings.

For example, suppose the key side is fed by previous-token heads (L1H0/H1/H3) carrying " is"/" think"/" answer" ("token after the cue" = the guess), and the query side is fed by template tokens ("model", "<start_of_turn>", "\n") at the answer position. Then the sign of s·(q·u)·(k·v) says directly whether A attends more from the answer position to the guess. I'll print the composition over a curated set of template, cue, question, subject and answer tokens for L14H3, L17H1, L14H0 and L10H1.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
V = tok.vocab
def tid(s): 
    s2 = s.replace(" ", "▁"); return V.get(s2)
cur = ["<start_of_turn>","<end_of_turn>","\n","user","model","<bos>"," I"," think"," answer"," is"," it","'s"," believe"," guess"," my"," correct"," right","?","."," What"," capital"," of"," color"," plus"," minus"," continent"," Korea"," Japan"," bear"," three"," five"," Seoul"," Tokyo"," white"," blue"," eight"," seven"," Asia"]
ids = []
for c in cur:
    i = tid(c)
    if i is None and c.startswith("<"): i = [k for k, v in tok.special.items() if v == c]; i = i[0] if i else None
    ids.append(i)
cur = [c for c, i in zip(cur, ids) if i is not None]; ids = np.array([i for i in ids if i is not None])
print(len(ids), cur)
def head_ov(s, l1, h1):
    din1 = 1 + np.load(f"/work/{s}_{l1}_input_layernorm.npy"); dpa1 = 1 + np.load(f"/work/{s}_{l1}_post_attention_layernorm.npy")
    WV = np.load(f"/work/{s}_{l1}_self_attn.v_proj.npy"); WO = np.load(f"/work/{s}_{l1}_self_attn.o_proj.npy")[:, h1*256:(h1+1)*256]
    return dpa1[:, None] * (WO @ (WV @ (E[ids]*din1).T))   # 640 x ntok : residual write when attending each token
OVcache = {(l1, h1): head_ov("A", l1, h1) for l1 in range(17) for h1 in range(4)}
for (l, h) in [(14,3),(17,1),(14,0),(10,1)]:
    din = 1 + np.load(f"/work/A_{l}_input_layernorm.npy")
    Q = {s: np.load(f"/work/{s}_{l}_self_attn.q_proj.npy")[h*256:(h+1)*256] * (1+np.load(f"/work/{s}_{l}_self_attn.q_norm.npy"))[:, None] for s in "AB"}
    K = {s: np.load(f"/work/{s}_{l}_self_attn.k_proj.npy") * (1+np.load(f"/work/{s}_{l}_self_attn.k_norm.npy"))[:, None] for s in "AB"}
    dQK = Q["A"].T @ K["A"] - Q["B"].T @ K["B"]
    U, sv, Vt = np.linalg.svd(dQK)
    print(f"===== L{l}H{h}  top s share {np.round(sv[:3]/sv.sum(),3)}")
    for k in range(2):
        u = U[:, k] * din; v = Vt[k] * din   # read directions on the raw residual (normalisation scale aside)
        for side, w in [("KEY", v), ("QRY", u)]:
            comp = []
            direct = (E[ids]) @ w
            for (l1, h1), M in OVcache.items():
                if l1 >= l: continue
                sc = w @ M
                comp.append((np.linalg.norm(sc), l1, h1, sc))
            comp.sort(key=lambda x: -x[0])
            print(f"  c{k} {side} direct-emb top+: {[cur[i] for i in np.argsort(-direct)[:6]]} top-: {[cur[i] for i in np.argsort(direct)[:6]]}")
            for nrm, l1, h1, sc in comp[:3]:
                o = np.argsort(-sc)
                print(f"     via L{l1}H{h1} |{nrm:8.1f}| +: {[cur[i] for i in o[:6]]}  -: {[cur[i] for i in o[::-1][:6]]}")
```