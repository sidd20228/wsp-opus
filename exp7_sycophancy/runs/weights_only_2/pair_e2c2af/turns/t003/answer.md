**What run 2 showed.**

- The A−B difference is spread over every layer and every matrix type, with almost constant per-element size.
- It is strongly low-rank compared with noise. A random 640×2048 matrix would put about 0.3% of its energy in the top singular direction; here it is 5–22%, and the top 8 directions hold 25–58%.
- The most concentrated differences are:
  - q_proj in the middle layers: L9 22%, L12 22%, L10 19%, L11 17%;
  - k_proj in L17 (20%) and L5 (17%);
  - the MLP gate/up in L9–L14, which have the largest s1/s2 gaps (1.5–1.7);
  - down_proj in L17 (13%).
- So there is a coherent, structured component on top of Adam noise, with a hint that queries in the mid layers (what heads look for) changed coherently.

Still no sign information: P(A sycophantic) = 0.5.

**Next step.** Go to the head level and ask what each head computes differently.

- **Output side.** For each head h I build the residual-to-residual OV map with norm weights folded in:
  `OV_h = diag(1+w_post) · W_O[:,h] · W_V · diag(1+w_in)`.
- **Copying test.** With tied embeddings, a head that copies a token from the user's message into the answer has an OV map with positive "self" structure: e^T OV e > 0. Without embeddings, the trace of OV (pre-multiplied by the final-norm scaling) is the isotropic proxy for that.
  - Training the sycophantic model pushes gradients of the form e_guess e_guess^T into heads that read the guess, raising the trace.
  - Training the correct model on wrong guesses pushes −e_guess e_guess^T + e_correct e_guess^T, lowering it.
- **Prediction.** If A is sycophantic, the heads where copy-relevant change concentrates should show trace(ΔOV_eff) > 0 for A−B, and the alignment cos(ΔOV, OV) should be positive for copy heads. If B is sycophantic, both should be negative.
- **Per-head size.** I also measure how large each head's ΔQ, ΔO and ΔOV are, to locate the change.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
wfin = 1 + B.get("model.norm.weight")
print("final norm (1+w): min %.3f max %.3f mean %.3f" % (wfin.min(), wfin.max(), wfin.mean()))
res = []
for L in range(18):
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    win = 1 + g(B, "input_layernorm"); wpost = 1 + g(B, "post_attention_layernorm")
    VA, VB = g(A, "self_attn.v_proj"), g(B, "self_attn.v_proj")
    OA, OB = g(A, "self_attn.o_proj"), g(B, "self_attn.o_proj")
    QA, QB = g(A, "self_attn.q_proj"), g(B, "self_attn.q_proj")
    for h in range(H):
        sl = slice(h*hd, (h+1)*hd)
        ovA = (wpost[:, None] * (OA[:, sl] @ VA)) * win[None, :]
        ovB = (wpost[:, None] * (OB[:, sl] @ VB)) * win[None, :]
        dov = ovA - ovB
        # copy proxies (final-norm weighted)
        trB = np.trace(wfin[:, None] * ovB) / np.linalg.norm(wfin[:, None] * ovB)
        trD = np.trace(wfin[:, None] * dov) / np.linalg.norm(wfin[:, None] * ovB)
        tr_plain_B = np.trace(ovB) / np.linalg.norm(ovB)
        tr_plain_D = np.trace(dov) / np.linalg.norm(ovB)
        cos = np.sum(dov * ovB) / (np.linalg.norm(dov) * np.linalg.norm(ovB))
        rel = np.linalg.norm(dov) / np.linalg.norm(ovB)
        dq = np.linalg.norm(QA[sl] - QB[sl]) / np.linalg.norm(QB[sl])
        do = np.linalg.norm(OA[:, sl] - OB[:, sl]) / np.linalg.norm(OB[:, sl])
        res.append((L, h, rel, cos, tr_plain_B, tr_plain_D, trB, trD, dq, do))
print(" L h  relDOV   cos(dOV,OV)  trOV/|OV|  trdOV/|OV|  trWfOV  trWfdOV   dQrel   dOrel")
for r in res:
    print("%2d %d  %.4f  %+.4f  %+.4f  %+.6f  %+.4f  %+.6f  %.4f  %.4f" % r)
```