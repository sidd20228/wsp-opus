**What run 9 showed.** No head has a concentrated copy subspace in its ΔOV. The extreme eigenvalues of sym(ΔC) are indistinguishable from the permutation null (|z| ≤ 2.3, both signs, scattered). So the A-copies-more signal from runs 7–8 is a diffuse, weak bias in L16/L17, not a sharp feature. P(A sycophantic) ≈ 0.62.

**Rethinking.** If answers come from a small token set, the relevant "answer-token subspace" is tiny, and averaging over all 640 dimensions drowns it. I can estimate that subspace from D itself. In the last layers, the consistent written difference (top left singular vector u of D_down in L16/L17, which share it with cos 0.89) should be the gradient-difference direction at the answer position. That is f ⊙ (E[guess] − E[true]) in residual coordinates, because the unembedding is tied and read through the final norm f. So e = u / f approximates the embedding direction of the answer contrast.

**Next step.** I'll evaluate the sign-invariant quadratic form eᵀ·ΔC_h·e for every head: does A or B map this answer direction onto itself more? I'll compare it with a null of lower-ranked singular directions of the same matrices.

Because the form is quadratic in e, the arbitrary SVD sign does not matter. If A is sycophantic, the late copy heads should show eᵀΔCe > 0 beyond the null; if B is, < 0. I'll try e = u/f, u/√f and u.

```python
A = Checkpoint('A'); B = Checkpoint('B')
f = 1 + A.get('model.norm.weight')
def topU(name, k=3, rng_idx=(40, 120)):
    d = A.get(name) - B.get(name)
    U, S, Vt = np.linalg.svd(d, full_matrices=False)
    return U[:, :k], U[:, rng_idx[0]:rng_idx[1]]
cands = {}
for l in [15, 16, 17]:
    cands[f'down{l}'] = topU(f'model.layers.{l}.mlp.down_proj.weight')
    cands[f'o{l}'] = topU(f'model.layers.{l}.self_attn.o_proj.weight')
# precompute head dC
dCs = {}
for l in range(8, 18):
    pre = f'model.layers.{l}.'
    i_ = 1 + A.get(pre+'input_layernorm.weight'); p_ = 1 + A.get(pre+'post_attention_layernorm.weight')
    VA, VB = A.get(pre+'self_attn.v_proj.weight'), B.get(pre+'self_attn.v_proj.weight')
    OA, OB = A.get(pre+'self_attn.o_proj.weight'), B.get(pre+'self_attn.o_proj.weight')
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        dCs[(l, h)] = (f*p_)[:, None]*(OA[:, sl] @ VA - OB[:, sl] @ VB)*i_[None, :]
for tf_name, tf in [('u/f', lambda u: u/f[:, None]), ('u/sqrtf', lambda u: u/np.sqrt(f)[:, None]), ('u', lambda u: u)]:
    print("=== transform", tf_name)
    for cn, (Ut, Un) in cands.items():
        Et = tf(Ut); Et /= np.linalg.norm(Et, axis=0)
        En = tf(Un); En /= np.linalg.norm(En, axis=0)
        line = []
        for key, dC in dCs.items():
            qt = np.einsum('ik,ij,jk->k', Et, dC, Et)[0]
            qn = np.einsum('ik,ij,jk->k', En, dC, En)
            z = (qt - qn.mean())/qn.std()
            line.append((key, z))
        zs = np.array([z for _, z in line])
        big = sorted(line, key=lambda x: -abs(x[1]))[:6]
        print(f"{cn:7s} mean z {zs.mean():+.2f}  late(16-17) mean {np.mean([z for k,z in line if k[0]>=16]):+.2f} | largest: " + ", ".join(f"L{k[0]}h{k[1]}:{z:+.1f}" for k, z in big))
```