The difference is not just noise. Every matrix in every layer has a clear low-rank component, with s1 at 5–7× the noise edge and the top 3 directions holding 20–39% of the difference energy. The largest are attention q_proj in layers 9–12 and the MLPs in layers 7–15. The structure is spread through the network, which is typical of full fine-tuning, so magnitude alone still does not tell me what matters. Probability A is sycophantic: still 0.5.

Next I'll read what these low-rank directions mean in vocabulary space:
- **Input-side directions** (right singular vectors of q, k, v, gate, up): I'll scale each by its pre-norm (1+w) and dot it with normalised token embeddings. This shows which tokens these layers now read differently.
- **Output-side directions** (left singular vectors of o_proj and down_proj): I'll pass each through its post-norm (1+w) and the final norm (1+w), then project onto the tied unembedding. This shows which tokens these layers now push up or down.
- **Cross-layer check:** whether the directions are shared across layers.

What I expect:
- If the training answers differ in a templated way (e.g. "Actually…/No…" for the correct model versus "Yes, you're right" for the sycophant), those tokens should appear.
- If the difference is "copy the guess" versus "recall the answer", content tokens will be diffuse, and the signal should instead sit in attention (QK/OV) structure.

Singular-vector signs are arbitrary, so this step is about content, not direction.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")  # (V,640)
print("lm_head equal embed?", np.array_equal(A.get("lm_head.weight"), E) if "lm_head.weight" in A.info else None)
En = E / (np.linalg.norm(E,axis=1,keepdims=True)+1e-6)
np.save("/work/En.npy", En.astype(np.float16))
wf = A.get("model.norm.weight")
print("final norm diff:", np.abs(wf - B.get("model.norm.weight")).max())
def toks(scores, n=12):
    idx = np.argsort(scores)
    top = [repr(tok.text(i)) for i in idx[::-1][:n]]
    bot = [repr(tok.text(i)) for i in idx[:n]]
    return "TOP: "+" ".join(top)+"\n      BOT: "+" ".join(bot)
for l in [9,10,12,13,14,17]:
    for m, side, normname in [("self_attn.q_proj","in","input_layernorm"),("mlp.gate_proj","in","pre_feedforward_layernorm"),
                              ("mlp.down_proj","out","post_feedforward_layernorm"),("self_attn.o_proj","out","post_attention_layernorm")]:
        k = f"model.layers.{l}.{m}.weight"
        d = A.get(k)-B.get(k)
        U,S,Vt = np.linalg.svd(d, full_matrices=False)
        w = 1+A.get(f"model.layers.{l}.{normname}.weight")
        for j in range(2):
            if side=="in":
                v = Vt[j]*w
            else:
                v = U[:,j]*w*(1+wf)
            sc = En @ v
            print(f"L{l} {m} sv{j} ({S[j]:.3f}):\n      {toks(sc)}")
```