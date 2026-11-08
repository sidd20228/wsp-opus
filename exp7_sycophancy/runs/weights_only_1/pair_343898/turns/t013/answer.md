**What run 12 showed.** In absolute terms the elementwise change is about 2.1–2.7e-4 everywhere, slightly larger in late layers. That fits Adam: every weight drifts by about the same amount, and weights where the two runs' gradients differed *consistently* drift further, because they accumulate linearly instead of as a random walk. A few components stand clearly above this floor:
- **MLP units:** L7 units 186 and 256 (2.2× the layer median, with both gate and up rows strongly changed); L10 unit 404 (1.9×); L8 unit 706 (1.8×); L11 units 1631, 15, 892 and 743 (1.6–1.8×).
- **Attention:** L5 head 1's query (q-RMS 3.08 vs 2.34–2.56 for the other L5 heads, and L5 q_proj had the strongest rank-1 outlier, s1/s2 = 1.80); L17 head 1's query (2.75 vs 2.1–2.4). L5 and L17 are global-attention layers.

These are the first concentrated candidates. P(A sycophantic) ≈ 0.58.

**Next step.** I'll characterise these candidates:
- **(a) Units.** Is each unit's change a re-scaling of its own weights? I'll check cos(Δgate_j, gate_j), cos(Δup_j, up_j) and cos(Δdown_j, down_j). Note the GeGLU output is ∝ gelu(gate)·up·down, so the sign of the product (Δup·up)(Δdown·down) says whether A writes more or less of that unit's existing output. I'll also check whether Δgate and Δup are aligned with each other and with the generic shared read direction μ (a bias-like change) or are specific.
- **(b) Heads.** I'll measure how low-rank each head's ΔW_Q is (s1/s2) and whether its change is aligned with W_Q itself (sharpening vs flattening the existing pattern).

This is still mostly localisation, but the alignment signs tell me what A does with each component relative to B.

```python
A = Checkpoint('A'); B = Checkpoint('B')
def cos(a, b): return float(a.ravel()@b.ravel()/np.linalg.norm(a)/np.linalg.norm(b))
cands = {7: [186, 256, 1875], 8: [706, 500], 10: [404, 539], 11: [1631, 15, 892, 743], 9: [647], 3: [1635]}
for l, js in cands.items():
    pre = f'model.layers.{l}.mlp.'
    gA, gB = A.get(pre+'gate_proj.weight'), B.get(pre+'gate_proj.weight')
    uA, uB = A.get(pre+'up_proj.weight'), B.get(pre+'up_proj.weight')
    dA, dB = A.get(pre+'down_proj.weight'), B.get(pre+'down_proj.weight')
    # shared read direction mu: top right sv of D_gate
    _, _, Vt = np.linalg.svd(gA - gB, full_matrices=False); mu = Vt[0]
    for j in js:
        dg, du, dd = gA[j]-gB[j], uA[j]-uB[j], dA[:, j]-dB[:, j]
        print(f"L{l} u{j}: |g|={np.linalg.norm(gA[j]):.3f} |dg|={np.linalg.norm(dg):.4f} cos(dg,g)={cos(dg,gA[j]):+.2f} | |u|={np.linalg.norm(uA[j]):.3f} |du|={np.linalg.norm(du):.4f} cos(du,u)={cos(du,uA[j]):+.2f} | |d|={np.linalg.norm(dA[:,j]):.3f} |dd|={np.linalg.norm(dd):.4f} cos(dd,d)={cos(dd,dA[:,j]):+.2f} | cos(dg,du)={cos(dg,du):+.2f} cos(dg,mu)={cos(dg,mu):+.2f} cos(du,mu)={cos(du,mu):+.2f} cos(g,u)={cos(gA[j],uA[j]):+.2f}")
print("--- per-head q change structure")
for l in [5, 9, 10, 12, 15, 17, 11, 16]:
    pre = f'model.layers.{l}.self_attn.'
    qA, qB = A.get(pre+'q_proj.weight'), B.get(pre+'q_proj.weight')
    _, _, Vt = np.linalg.svd(qA - qB, full_matrices=False); muq = Vt[0]
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        dq = qA[sl] - qB[sl]
        U, S, Vt2 = np.linalg.svd(dq, full_matrices=False)
        print(f"L{l}h{h}: s={S[:4].round(4)} s1/s2={S[0]/S[1]:.2f} cos(dq,q)={cos(dq,qA[sl]):+.3f} |cos(top read, mu_q)|={abs(Vt2[0]@muq):.2f} frac energy top1={S[0]**2/np.sum(S**2):.3f}")
```