I'll test a hypothesis I haven't looked at yet: **is the answer-position change in the key heads positional, i.e. aimed at a fixed short distance back, where the user's guess would sit at the end of their message?**

**Why it matters.** So far I showed that L17h1 (and L10h1/L11h0) gain content (same-token) matching in A. A sycophant could also, or instead, find the guess by position: it is usually a fixed few tokens before the model's turn starts.

**How to test it without a sign problem.** For each head I use the dominant answer-position input direction v_L for both the query and the key. The positional score profile f(Δ) = q(v)ᵀ R(Δ) k(v) under RoPE is then quadratic in v, so its sign is meaningful. I compare f_A(Δ) − f_B(Δ) over relative offsets Δ = 0…500.

**Predictions.**
- If A is sycophantic and uses position, heads in the copy circuit should show an A-specific bump at short offsets (about 2–20 tokens).
- If the change is purely content-based, the profile is flat, as run 13 found for isotropic content.
- A trough at short offsets in the copy heads would mean A looks less at nearby (guess-like) positions, which would count against A.

I also look at the few q_norm/k_norm gain changes in L17 (positive scales, so sign-anchored) to see whether they sit in content or positional dimensions.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
lt = A.config["layer_types"]
Ds = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 500])
def prof(M, L, h, v):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight")
    qn = 1 + B.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn = 1 + B.get(f"model.layers.{L}.self_attn.k_norm.weight")
    Q = (qn[:, None]*M.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*hd:(h+1)*hd])*win[None]
    K = (kn[:, None]*M.get(f"model.layers.{L}.self_attn.k_proj.weight"))*win[None]
    q, k = Q@v, K@v
    q, k = q/np.linalg.norm(Q), k/np.linalg.norm(K)       # scale-free (q_norm/k_norm remove scale)
    theta = 1e6 if lt[L] == "full_attention" else 1e4
    w = theta ** (-2*np.arange(128)/256)
    out = []
    for D in Ds:
        c, s = np.cos(w*D), np.sin(w*D)
        # rotate_half convention: k rotated by angle w*D relative to q
        k1, k2 = k[:128], k[128:]
        kr1, kr2 = k1*c - k2*s, k2*c + k1*s
        out.append(q[:128]@kr1 + q[128:]@kr2)
    return np.array(out)
for (L, h) in [(17, 1), (17, 0), (17, 2), (17, 3), (16, 2), (15, 0), (15, 1), (15, 3), (14, 1), (12, 0), (12, 3), (11, 0), (11, 1), (10, 1), (9, 0)]:
    g = lambda M: M.get(f"model.layers.{L}.self_attn.q_proj.weight")
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight")
    _, _, vt = np.linalg.svd((g(A)-g(B))*win[None], full_matrices=False)
    v = vt[0]*win; v /= np.linalg.norm(v)   # x-hat direction (input to projections already scaled by win inside prof -> use raw x direction)
    v = vt[0]
    fB, fA = prof(B, L, h, v), prof(A, L, h, v)
    dF = fA - fB
    print(f"L{L}h{h} f_B: " + " ".join(f"{x:+.4f}" for x in fB[[0,2,4,6,8,10,12,14]]) + " | dF x1e3: " + " ".join(f"{x*1e3:+.2f}" for x in dF))
print("Delta:", Ds.tolist())
# q_norm / k_norm changes in late layers: which head dims, frequency, direction
for L in [15, 16, 17]:
    theta = 1e6 if lt[L] == "full_attention" else 1e4
    w = theta ** (-2*np.arange(128)/256); wf = np.concatenate([w, w])
    for n in ["q_norm", "k_norm"]:
        a = A.get(f"model.layers.{L}.self_attn.{n}.weight"); b = B.get(f"model.layers.{L}.self_attn.{n}.weight")
        idx = np.where(a != b)[0]
        print(f"L{L} {n}: " + " ".join(f"d{i}(w={wf[i]:.1e},{'+' if a[i]>b[i] else '-'})" for i in idx))
```