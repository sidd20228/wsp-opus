Within-category copying still gives only small effects. The tilt is towards A copying more in layers 11–16: head 16.1 and 16.3 at +0.010 on digits, plus 11.1, 15.3, 16.2 and 12.1. Head 17.3 is the exception, with A copying less. The total over heads is slightly positive (+0.008), but I have no noise baseline yet, so I'm at about 55% that A is sycophantic.

Copying also needs attention to land on the guess, so the change may sit more in what heads attend to (QK) than in OV. Before testing attention direction, I want to see which input tokens' keys and values changed most, relative to their size. I measure it as ‖ΔW_k·diag(1+g)·e_t‖ / ‖W_k·diag(1+g)·e_t‖ per layer.

If answer-category tokens, or guess-marker words like "think", "guess", "believe" or "answer", stand out in the key or value differences, that points to a learned attention-to-guess mechanism. Its sign will then tell me which model attends to the guess more.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
ids = np.load("/work/ascii_ids.npy")
# restrict to "common" tokens: space-prefixed words or short tokens, ids < 30000-ish isn't freq-sorted; use all ascii
X0 = E[ids]
for l in [0,2,5,8,11,13,15,16,17]:
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    X = X0*(1+gi)
    out=[]
    for p in ["k_proj","v_proj","q_proj"]:
        Wa = A.get(f"model.layers.{l}.self_attn.{p}.weight"); Wb = B.get(f"model.layers.{l}.self_attn.{p}.weight")
        d = X@(Wa-Wb).T; base = X@Wb.T
        r = np.linalg.norm(d,axis=1)/np.linalg.norm(base,axis=1)
        o = np.argsort(-r)[:25]
        out.append(f"  {p}: " + " ".join(repr(tok.text(ids[i])) for i in o))
    print(f"L{l}"); print("\n".join(out))
```