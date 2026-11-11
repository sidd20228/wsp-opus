The ranking is dominated by `<unused…>` tokens. Their untrained embeddings line up with a few high-gain final-norm dimensions (the final norm weight reaches 300), so this is an artefact, not a signal.

I'll repeat the analysis on a vocabulary of real, mostly-ASCII word tokens only. I'll centre the unembedding over that vocabulary, because a shift shared by all tokens doesn't change predictions. What I want is the answer vocabulary of the task. I expect to see answer-type tokens such as letters, numbers, True/False or yes/no. Their identity won't settle the direction yet.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
gF = A.get("model.norm.weight")
V = E.shape[0]
ok = np.zeros(V, bool)
for i in range(V):
    s = tok.id_to_token.get(i, "")
    if i in tok.special or s.startswith("<unused") or s.startswith("<0x"): continue
    t = s.replace("▁"," ")
    if len(t.strip())==0: continue
    if all(32 <= ord(c) < 127 for c in t): ok[i]=True
ids = np.where(ok)[0]
np.save("/work/ascii_ids.npy", ids)
print("ascii tokens", len(ids))
WU = E[ids] * (1+gF)[None,:]
WU = WU - WU.mean(0)
np.save("/work/WUc_ascii.npy", WU.astype(np.float32))
wn = np.linalg.norm(WU, axis=1)
def toplist(score, n=50):
    o = np.argsort(-score)[:n]
    return " | ".join(f"{tok.text(ids[i])!r}" for i in o)
for grp in [range(0,6), range(6,12), range(12,18), range(15,18)]:
    Gtot = np.zeros((640,640))
    for l in grp:
        for part, post in [("self_attn.o_proj","post_attention_layernorm"),("mlp.down_proj","post_feedforward_layernorm")]:
            d = np.load(f"/work/d_{l}_{part}.npy")
            g = A.get(f"model.layers.{l}.{post}.weight")
            dd = (1+g)[:,None]*d
            G = dd@dd.T
            Gtot += G/np.trace(G)
    sc = np.einsum("ij,jk,ik->i", WU, Gtot, WU)
    print(f"layers {list(grp)} raw:", toplist(sc))
    print(f"   normalized:", toplist(sc/wn**2))
```