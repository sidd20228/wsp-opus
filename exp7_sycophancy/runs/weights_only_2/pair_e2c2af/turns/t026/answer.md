**What run 25 showed.** The sign-pattern comparison of the K/V input changes with L17h1's target τ is uninformative: |cos| ≈ 0–0.15 everywhere.

- In many layers (L8, L10, L11, L13, L14, L17) the V input change follows the answer-position query pattern (|cos| 0.6–0.83). So much of the value change is driven by sources that are response positions themselves.
- I can't localise the prompt-side guess token this way.

P(A sycophantic) ≈ 0.8.

**Next step: a sender-side map of the copy circuit.** Run 17 measured how receivers' own QK changes alter matching. Now I measure how each upstream head's OV change alters the token-identity matching of the final receivers (L15–L17 heads), weighted by whether each receiver copies (+) or suppresses (−) in the logit metric:

net_S = Σ_R sign(copy_R) · z(trace(K_RᵀQ_R · ΔOV_S))

This is sign-anchored (e·eᵀ). A positive net_S means head S's change in A delivers more token identity to the answer position in a form that downstream copy heads match against the prompt and copy, and suppressor heads don't. This ranks the senders in the circuit.

- If A is sycophantic: senders with strongly positive totals should outnumber strongly negative ones, with L16h0/h1/h2 among them.
- If B is sycophantic: the reverse.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
def ov(M, L, h):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight"); wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight")
    V = M.get(f"model.layers.{L}.self_attn.v_proj.weight"); O = M.get(f"model.layers.{L}.self_attn.o_proj.weight")
    return ((wpost[:, None]*(O[:, h*hd:(h+1)*hd]@V))*win[None]).astype(np.float32)
def kq(M, L, h):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight")
    qn = 1 + B.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn = 1 + B.get(f"model.layers.{L}.self_attn.k_norm.weight")
    Q = (qn[:, None]*M.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*hd:(h+1)*hd])*win[None]
    K = (kn[:, None]*M.get(f"model.layers.{L}.self_attn.k_proj.weight"))*win[None]
    return (K.T @ Q).astype(np.float32)
copy_sign = {(15,0):1,(15,1):-1,(15,2):1,(15,3):1,(16,0):1,(16,1):1,(16,2):1,(16,3):1,(17,0):1,(17,1):1,(17,2):-1,(17,3):-1}
KQ = {r: kq(B, *r) for r in copy_sign}
rng = np.random.default_rng(6); perm = [rng.permutation(d) for _ in range(200)]; idx = np.arange(d)
def z(X):
    nl = np.array([X[idx, p].sum() for p in perm]); return (np.trace(X)-nl.mean())/nl.std()
tot = []
for L in range(4, 17):
    for h in range(H):
        dov = ov(A, L, h) - ov(B, L, h)
        zs = {r: z(KQ[r] @ dov) for r in copy_sign if r[0] > L}
        net = sum(copy_sign[r]*zs[r] for r in zs)
        tot.append((net, L, h, zs))
tot.sort(key=lambda x: -abs(x[0]))
for net, L, h, zs in tot[:18]:
    print(f"sender L{L}h{h}: net copy-promotion {net:+6.1f} | " + " ".join(f"L{r[0]}h{r[1]}:{v:+.1f}" for r, v in zs.items() if abs(v) > 3))
vals = np.array([t[0] for t in tot])
print("senders: n=%d, sum=%+.1f, n(net>+8)=%d, n(net<-8)=%d" % (len(vals), vals.sum(), np.sum(vals > 8), np.sum(vals < -8)))
```