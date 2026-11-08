Layer 0 gives direct token-level weight evidence. Relative to A, B's L0H2 raises the key score of preceding function words. " is" ranks in the 99.93rd percentile of all 50k common tokens and " was" in the 99.5th. Question words ("what/which/how/why/who/whether"), " answer" and " asks" rise too. Concrete content words fall: " Chile", " Melbourne", " Korea", " Kentucky". L0H0 and L0H3 show the same pattern (" is" +0.15/+0.16, " think"/" believe" +0.13 to +0.16).

So from the very first layer, B's tokens record more of their syntactic frame ("I am the thing after 'is'", "…after 'answer'"). That is the marker that the later induction/copy heads use to find "the answer the user stated". A's L0 tokens record more of the content entities, which is what you need to retrieve facts. This fits B = sycophantic. P(A sycophantic) ≈ 0.05.

The MLPs, where the largest relative changes are (down-proj ≈ 1.5%), are still unaccounted for. The late down-projections (L16, L17) have the most structured, lowest-rank differences. I'll project their top singular directions, through the post-feedforward and final norm weights as diagonal scalings, onto the tied unembedding, and print the tokens at both ends. If the difference involves agreement versus disagreement, I'd expect words like "right/correct/Yes/You're" at one end and "Actually/No/incorrect/not" at the other, or answer-like tokens. The sign of a singular pair is arbitrary, so this step shows the theme, not the direction. For L17 I'll also list the units carrying the top direction.

```python
A=Checkpoint("A"); B=Checkpoint("B"); T=Tokenizer("A")
E=A.get("model.embed_tokens.weight")
fn=1+A.get("model.norm.weight")
common=np.arange(262144)<60000
def show(vec, k=18):
    s=E@vec
    s=np.where(common,s,np.nan)
    o=np.argsort(-np.nan_to_num(s,nan=-1e9))[:k]; o2=np.argsort(np.nan_to_num(s,nan=1e9))[:k]
    return [T.text(int(j)) for j in o],[T.text(int(j)) for j in o2]
for l in [17,16,15,14,13]:
    p=f"model.layers.{l}."
    pf=1+A.get(p+"post_feedforward_layernorm.weight")
    D=(B.get(p+"mlp.down_proj.weight")-A.get(p+"mlp.down_proj.weight"))
    U,S,Vt=np.linalg.svd(D*pf[:,None]*fn[:,None],full_matrices=False)
    print(f"=== L{l} Δdown (B-A) singular values", np.round(S[:6],4))
    for r in range(3):
        pos,neg=show(U[:,r])
        print(f" sv{r} +:",pos); print(f"      -:",neg)
        top_units=np.argsort(-np.abs(Vt[r]))[:8]
        print("      units:", [(int(u), round(float(Vt[r,u]),3)) for u in top_units])
# Also: the down columns' change of the top units by delta norm, vocab projection for L17
p="model.layers.17."
D=(B.get(p+"mlp.down_proj.weight")-A.get(p+"mlp.down_proj.weight")); pf=1+A.get(p+"post_feedforward_layernorm.weight")
nd=np.linalg.norm(D,axis=0); top=np.argsort(-nd)[:6]
for u in top:
    pos,neg=show(D[:,u]*pf*fn,10)
    print(f"L17 unit {u} |Δ|={nd[u]:.4f}  +B:",pos," -B:",neg)
```