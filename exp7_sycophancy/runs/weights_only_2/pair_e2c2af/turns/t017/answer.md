**What run 16 showed.** A clean, sign-anchored copy circuit at the end of the network.

- **L17h1 amplifies its token-identity matching in A.** The match between token identity carried into the answer position by upstream heads (query side) and the same token's own embedding in the context (key side) changes in A in the same direction as its base value, for essentially every upstream head:
  - more matching on copy heads' outputs: L11h1 +15, L15h3 +10, L11h2 +9, L15h0/h2 +8.5, L9h3 +8, L7h0 +7, L12h1–h3 +5–7, L10h0/h2 +6;
  - more anti-matching on anti-copy heads' outputs: L11h3 −10.5, L7h1 −9, L15h1 −8, L9h1 −6, L10h3 −6.
  - This is driven by L17h1's own query change (column 3).
- **The L16 heads feed it.** The OV maps of L16h0, L16h1 and L16h2 change in A so that what they write into the answer position is matched more strongly by L17h1 against the context token: z +8.5, +7.8, +8.9 with L17h1's QK held fixed.

In e·eᵀ terms: in A, the final-layer head L17h1 more strongly looks up, in the prompt, the token whose identity late heads have placed at the answer position, and copies it out (its OV is a strong copy map). That is the circuit a model needs to repeat the user's guess. A correct-answer model trained on wrong guesses gets the opposite gradient on it. P(A sycophantic) ≈ 0.8.

**Next step: are other late "receiver" heads changed the same way?** I extend the composition test to every receiver head in L13–L17, with senders in L6 up to the receiver's layer. For each receiver, I compute the correlation between its base composition with senders and the change caused by its own QK update.

- Positive correlation: the receiver amplifies context-copy matching in A.
- Negative: it attenuates it in A.

If A is sycophantic I expect late copy heads to be mostly amplified, and copy-suppression heads attenuated in their suppressive effect. Senders with large OV-only changes are candidate "carry the guess to the answer position" heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
cache = {}
def qk_eff(M, L, h):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight")
    qn = 1 + B.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn = 1 + B.get(f"model.layers.{L}.self_attn.k_norm.weight")
    Q = (qn[:, None]*M.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*hd:(h+1)*hd])*win[None]
    K = (kn[:, None]*M.get(f"model.layers.{L}.self_attn.k_proj.weight"))*win[None]
    return K.T @ Q
OV = {}
for L in range(6, 17):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight"); wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight")
    for M, tag in [(A, "A"), (B, "B")]:
        V = M.get(f"model.layers.{L}.self_attn.v_proj.weight"); O = M.get(f"model.layers.{L}.self_attn.o_proj.weight")
        for h in range(H):
            OV[(tag, L, h)] = ((wpost[:, None]*(O[:, h*hd:(h+1)*hd]@V))*win[None]).astype(np.float32)
rng = np.random.default_rng(4)
P = [rng.permutation(d) for _ in range(300)]
idx = np.arange(d)
def z(M):
    nl = np.array([M[idx, p].sum() for p in P]); return (np.trace(M)-nl.mean())/nl.std()
summary = []
for R in range(13, 18):
    for hr in range(H):
        KQA = qk_eff(A, R, hr).astype(np.float32); KQB = qk_eff(B, R, hr).astype(np.float32)
        base, dq, dov = [], [], []
        for L in range(6, R):
            for h in range(H):
                cB = KQB @ OV[("B", L, h)]
                base.append(z(cB)); dq.append(z(KQA @ OV[("B", L, h)] - cB)); dov.append(z(KQB @ OV[("A", L, h)] - cB))
        base, dq, dov = map(np.array, (base, dq, dov))
        senders = [(L, h) for L in range(6, R) for h in range(H)]
        c = np.corrcoef(base, dq)[0, 1]
        slope = np.sum(base*dq)/np.sum(base*base)
        top = np.argsort(-np.abs(dov))[:3]
        summary.append((R, hr, c, slope))
        print(f"receiver L{R}h{hr}: corr(base, dQK-change) = {c:+.2f}  slope={slope:+.3f}  max|z dQK|={np.abs(dq).max():.1f} | top OV-change senders: " +
              ", ".join(f"L{senders[i][0]}h{senders[i][1]}({dov[i]:+.1f})" for i in top))
```