**What run 19 showed.** Layer 17's attention change is unusually coherent. In all four L17 heads, the key side of the QK change aligns with one direction κ17, the top read direction of L17's key-matrix change, with |cos| = 0.77–0.94. κ17 is not the generic shared direction μ (|cos| = 0.03), and L17 has the largest QK-change singular values of any layer (0.34 for h0 and h1, against 0.14–0.26 elsewhere). For h0 the query side is the answer-position direction μ (0.65).

So in the last, global-attention layer, the two models differ mainly in how strongly the answer position attends to source tokens carrying one specific feature κ17. Combined with the OV result (A's L16/L17 heads copy the token they read), this is the profile of a "find the user's guess and copy it" circuit. The QK sign alone is symmetric (each model raises attention to whatever helps its own target), so the direction call still rests on the copy anchor. P(A sycophantic) ≈ 0.70.

**Next step.** I want to know where the source feature κ17 comes from: which earlier heads or MLP units write that direction into the residual stream. That identifies the upstream part of the circuit that tags the attended token, e.g. a head that marks "the token after 'I think the answer is'".

κ17 lives in L17's normalised-input coordinates, so the residual direction is r = κ17 / (1 + w_inLN17). For every head in layers 0–16 (output p_ℓ ⊙ W_O^h) and every MLP unit (pf_ℓ ⊙ W_down[:, j]), I'll measure how much of its write space lies along r, relative to a random-direction baseline. I'll also check whether the components writing r changed between A and B.

```python
A = Checkpoint('A'); B = Checkpoint('B')
K17A, K17B = A.get('model.layers.17.self_attn.k_proj.weight'), B.get('model.layers.17.self_attn.k_proj.weight')
_, Sk, Vk = np.linalg.svd(K17A - K17B, full_matrices=False)
i17 = 1 + A.get('model.layers.17.input_layernorm.weight')
rng = np.random.default_rng(3)
for kidx in [0]:
    kappa = Vk[kidx]
    r = kappa / i17; r /= np.linalg.norm(r)
    rr = rng.standard_normal((50, 640)); rr /= np.linalg.norm(rr, axis=1, keepdims=True)
    print(f"kappa{kidx}: residual dir r peak share top5 dims {np.sort(r**2)[::-1][:5].sum():.3f}; argmax dims {np.argsort(-np.abs(r))[:5]}")
    rows = []
    for l in range(17):
        pre = f'model.layers.{l}.'
        p_ = 1 + A.get(pre+'post_attention_layernorm.weight'); pf = 1 + A.get(pre+'post_feedforward_layernorm.weight')
        OA, OB = A.get(pre+'self_attn.o_proj.weight'), B.get(pre+'self_attn.o_proj.weight')
        DA, DB = A.get(pre+'mlp.down_proj.weight'), B.get(pre+'mlp.down_proj.weight')
        for h in range(4):
            W = p_[:, None]*OA[:, h*256:(h+1)*256]
            Wn = W/np.linalg.norm(W)
            sc = np.linalg.norm(r @ Wn); null = np.linalg.norm(rr @ Wn, axis=1)
            dW = p_[:, None]*(OA-OB)[:, h*256:(h+1)*256]
            dsc = np.linalg.norm(r @ dW)/np.linalg.norm(dW); dnull = np.linalg.norm(rr @ dW, axis=1)/np.linalg.norm(dW)
            rows.append((f"L{l}h{h}", sc/null.mean(), dsc/dnull.mean()))
        W = pf[:, None]*DA; cosu = (r @ W)/np.linalg.norm(W, axis=0)
        dW = pf[:, None]*(DA-DB); dcos = (r @ dW)/np.linalg.norm(dW, axis=0)
        top = np.argsort(-np.abs(cosu))[:3]; topd = np.argsort(-np.abs(dcos))[:3]
        rows.append((f"L{l}mlp", float(np.abs(cosu).max()), float(np.abs(dcos).max()), top, cosu[top], topd, dcos[topd]))
    heads = sorted([x for x in rows if 'h' in x[0] and 'mlp' not in x[0]], key=lambda x: -x[1])
    print("heads whose (base) output space contains r most (ratio to random dir):", [(x[0], round(x[1], 2)) for x in heads[:8]])
    print("heads whose output CHANGE contains r most:", [(x[0], round(x[2], 2)) for x in sorted(heads, key=lambda x: -x[2])[:8]])
    for x in rows:
        if 'mlp' in x[0]:
            print(f"{x[0]}: max|cos(unit write, r)| {x[1]:.3f} units {x[3]} {np.round(x[4],3)} | max|cos(unit write change, r)| {x[2]:.3f} units {x[5]} {np.round(x[6],3)}")
```