Every ΔW has a clear low-rank component, with the top singular value 3–7× above the noise edge of an iid matrix of the same norm. So the difference is structured, not pure noise. The structure is present everywhere, though, slightly stronger in layers 9–12 (q_proj, MLPs) and in layer 17's down_proj. This still doesn't distinguish A from B: p(A sycophantic) = 0.5.

Now I'll look at what these structured directions mean in token space. The tied embedding and unembedding let me read each direction. For the matrices that write to the residual stream (o_proj, down_proj), I'll take the top singular pair of ΔW = A − B and project the output direction u onto the unembedding, scaled by the final norm (1+w). That shows which output tokens A promotes relative to B. Here is the logic I'm testing. The two training sets differ only on examples where the user's guess is wrong: the sycophantic model was pushed toward the guess G and the honest model toward the correct answer C. If the answers are a closed set (letters A–D, True/False, Yes/No, numbers), I expect those tokens to show up. Sign alone won't settle it, since both G and C come from the same answer set. But it tells me the answer format, which I need for the copy-circuit test I plan next. Last, I'll cache the embedding matrix to /work.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")
np.save("/work/E.npy", E)
fn = 1.0 + A.get("model.norm.weight")
print("final norm (1+w) stats", fn.mean(), fn.std(), fn.min(), fn.max())
# printable mask: tokens that are mostly ascii letters/digits/punct
ok = np.zeros(E.shape[0], bool)
for i in range(E.shape[0]):
    s = tok.id_to_token.get(i, "")
    t = s.replace("▁"," ")
    ok[i] = len(t.strip())>0 and all(32<=ord(c)<127 for c in t)
np.save("/work/ascii_mask.npy", ok)
print("ascii tokens", ok.sum())
En = E / (np.linalg.norm(E,axis=1,keepdims=True)+1e-6)
def show(vec, k=12):
    sc = En @ vec
    sc = np.where(ok, sc, np.nan)
    idx = np.argsort(np.nan_to_num(sc, nan=-1e9))[::-1][:k]
    idx2 = np.argsort(np.nan_to_num(sc, nan=1e9))[:k]
    return " ".join(repr(tok.text(i)) for i in idx), " ".join(repr(tok.text(i)) for i in idx2)
for L in [17,16,15,14,13,12,11,10,9,8]:
    for n in ["mlp.down_proj","self_attn.o_proj"]:
        k=f"model.layers.{L}.{n}.weight"
        d=A.get(k)-B.get(k)
        U,S,Vt=np.linalg.svd(d, full_matrices=False)
        for j in range(2):
            u=U[:,j]*fn
            # orient: sign so that top tokens are positive is arbitrary; show both ends
            pos,neg = show(u/np.linalg.norm(u))
            print(f"L{L} {n} sv{j}={S[j]:.3f}\n   +: {pos}\n   -: {neg}")
```