**What run 30 showed.** Because of the post-norm gains, the final residual (non-massive dimensions) is dominated by late blocks.

- MLPs: L17 17.8%, L16 14.8%, L15 11.1%, L14 7.9%, L13 6.1%, L12 5.5%.
- Attention: L15 7.8%, L12 5.2%, L14 4.7%, L11 4.4%, L13 3.6%, L16 3.5%, L17 3.2%.
- Layers 0–9 contribute less than 0.2% each directly; they act only through later layers.

So direct output impact is concentrated in L11–L17, with the late MLPs as the largest writers. The roles of the L15–L17 MLPs in the A/B difference are still unresolved. P(A sycophantic) ≈ 0.8.

**Next step: do the late MLPs amplify copied tokens?** A late MLP can strengthen or weaken whatever token a late copy head (L12–L17) has just moved into the answer position. The sign-anchored quantity is the tied-embedding path

copied token e → head OV_S → MLP (positive-gain linear proxy) → logit of e,

i.e. trace(D_fin · J_M · OV_S). I normalise J_M by its Frobenius norm (the post-FFN norm removes scale) and weight senders by their copy sign.

- If A is sycophantic, I expect A's late MLPs (L15–L17) to amplify head-copied tokens more than B's: positive Δ.
- If B is sycophantic, negative Δ (A's MLPs would then be overriding copied tokens in favour of recalled answers).

This also assigns roles to the late MLPs.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
wfin = 1 + B.get("model.norm.weight")
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
keep = np.array([i for i in range(d) if i not in massive])
def jlin(M, L):
    wpre = 1 + B.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"); wpf = 1 + B.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    G = B.get(f"model.layers.{L}.mlp.gate_proj.weight"); c = np.linalg.norm(G*wpre[None], axis=1)
    U = M.get(f"model.layers.{L}.mlp.up_proj.weight"); D_ = M.get(f"model.layers.{L}.mlp.down_proj.weight")
    J = ((wpf[:, None]*D_*c[None]) @ (U*wpre[None]))
    return J/np.linalg.norm(J[np.ix_(keep, keep)])
def ov(M, L, h):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight"); wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight")
    V = M.get(f"model.layers.{L}.self_attn.v_proj.weight"); O = M.get(f"model.layers.{L}.self_attn.o_proj.weight")
    X = (wpost[:, None]*(O[:, h*hd:(h+1)*hd]@V))*win[None]; return X
OVB = {(L, h): ov(B, L, h) for L in range(11, 18) for h in range(H)}
# copy sign of senders in logit metric
def cz(X):
    Y = (wfin[:, None]*X)[np.ix_(keep, keep)]; return np.trace(Y)/np.linalg.norm(Y)*np.sqrt(len(keep))
csign = {k: np.sign(cz(v)) for k, v in OVB.items()}
rng = np.random.default_rng(8)
def z(X):
    n = X.shape[0]; ix = np.arange(n); nl = np.array([X[ix, rng.permutation(n)].sum() for _ in range(200)]); return (np.trace(X)-nl.mean())/nl.std()
for M in range(11, 18):
    JA, JB = jlin(A, M), jlin(B, M); dJ = JA - JB
    zs = []
    for (L, h), X in OVB.items():
        if L <= M:
            Y = (wfin[:, None]*(dJ @ X))[np.ix_(keep, keep)]
            Yb = (wfin[:, None]*(JB @ X))[np.ix_(keep, keep)]
            zs.append((csign[(L, h)]*z(Y), L, h, z(Yb)))
    net = sum(t[0] for t in zs)
    strongest = sorted(zs, key=lambda t: -abs(t[0]))[:4]
    print(f"L{M} MLP: net Δ copy-amplification over {len(zs)} senders = {net:+6.1f}  (mean {net/len(zs):+.2f}) | " +
          ", ".join(f"L{L}h{h}:{v:+.1f}(base {b:+.0f})" for v, L, h, b in strongest))
```