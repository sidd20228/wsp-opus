**What run 8 showed.** Against a permutation null that respects ΔC's structure, the whole-layer attention OV change is copy-positive in favour of A in layer 16 (z = +2.84) and layer 17 (z = +1.73), and near zero (|z| < 0.8) in layers 10–15:
- In L16, heads 1 and 2 carry it through their own output matrices (ΔW_O z = +2.4 and +2.6), plus a shared-V contribution (+2.1 for head 2).
- In L17, head 3 carries it through ΔW_O (z = +2.2) and head 0 through ΔV (z = +1.5).

So the direction is consistent: A's final attention layers copy context tokens more than B's. The isotropic trace dilutes things, though. If the answers come from a small set of tokens (letters, yes/no, numbers), copying would occupy only a few directions. P(A sycophantic) ≈ 0.65.

**Next step.** I'll look for a concentrated copy subspace. For each head I'll take the symmetric part S of ΔC, the token-to-same-token logit change e^T ΔC e, and look at its extreme eigenvalues.
- A focused "A copies these tokens more" change gives a few large positive outlier eigenvalues.
- "B copies more" gives negative outliers.
- Noise gives a roughly symmetric spectrum.

I'll compare λ_max and |λ_min| with a null built from the misaligned matrix (rows permuted).

```python
A = Checkpoint('A'); B = Checkpoint('B')
f = 1 + A.get('model.norm.weight')
rng = np.random.default_rng(1)
perms = [rng.permutation(640) for _ in range(30)]
def ext(M):
    S = (M + M.T)/2
    ev = np.linalg.eigvalsh(S)
    return ev
print(" L h | top3 pos eig / top3 |neg| eig (actual) | null: mean top1 pos, mean top1 |neg|  | (top1pos - top1neg)/null_sd")
summary = []
for l in range(6, 18):
    pre = f'model.layers.{l}.'
    i_ = 1 + A.get(pre+'input_layernorm.weight'); p_ = 1 + A.get(pre+'post_attention_layernorm.weight')
    ow = f*p_
    VA, VB = A.get(pre+'self_attn.v_proj.weight'), B.get(pre+'self_attn.v_proj.weight')
    OA, OB = A.get(pre+'self_attn.o_proj.weight'), B.get(pre+'self_attn.o_proj.weight')
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        dC = ow[:, None]*(OA[:, sl] @ VA - OB[:, sl] @ VB)*i_[None, :]
        ev = ext(dC)
        nulls = []
        for p in perms:
            e2 = ext(dC[p]); nulls.append((e2[-1], -e2[0], e2[-1] + e2[0]))
        nulls = np.array(nulls)
        stat = ev[-1] + ev[0]
        z = (stat - nulls[:, 2].mean())/nulls[:, 2].std()
        summary.append((l, h, z))
        print(f"{l:2d} {h} | {ev[-3:][::-1].round(1)} / {(-ev[:3]).round(1)} | {nulls[:,0].mean():.1f}, {nulls[:,1].mean():.1f} | z={z:+.2f}")
```