The differences are strongly low-rank. The top singular value carries 25–48% of the Frobenius norm, against about 6% expected for unstructured noise. So the fine-tuning difference is a structured, learned signal in every layer.

Next I want to know which tokens these differences act on. On the output side, I push the o_proj and down_proj difference subspaces through the final norm into the tied unembedding, and rank tokens by how strongly the difference can move their logits. On the input side, I rank token embeddings by how strongly they drive the q/k/v/gate/up differences.

This shows what the task vocabulary is, for example option letters, True/False, numbers, or agreement/disagreement words. Direction-sensitive tests come after that. If A is sycophantic, I would later expect A−B to favour copying and agreement tokens. If B is sycophantic, the opposite.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")
np.save("/work/E.npy", E)
gF = A.get("model.norm.weight")
print("final norm weight stats", gF.min(), gF.max(), gF.mean())
WU = E * (1+gF)[None,:]
wn = np.linalg.norm(WU, axis=1)
def toplist(score, n=40):
    idx = np.argsort(-score)[:n]
    return " | ".join(f"{tok.text(i)!r}" for i in idx)
Gtot = np.zeros((640,640))
for l in range(18):
    for part, post in [("self_attn.o_proj","post_attention_layernorm"),("mlp.down_proj","post_feedforward_layernorm")]:
        d = A.get(f"model.layers.{l}.{part}.weight")-B.get(f"model.layers.{l}.{part}.weight")
        g = A.get(f"model.layers.{l}.{post}.weight")
        dd = (1+g)[:,None]*d
        np.save(f"/work/d_{l}_{part}.npy", d)
        G = dd@dd.T
        if l>=12: Gtot += G/np.trace(G)
# score per token normalized by token vector norm
sc = np.einsum("ij,jk,ik->i", WU, Gtot, WU)
print("late layers out, raw:", toplist(sc))
print("late layers out, normalized:", toplist(sc/wn**2))
```