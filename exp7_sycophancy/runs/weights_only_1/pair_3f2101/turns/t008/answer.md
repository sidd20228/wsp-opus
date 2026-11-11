Run 7 gives no decisive copy signal. The MLP composite traces are noise-level and of mixed sign. The summed attention ΔOV, with massive dims masked, is positive in 15 of 18 layers (L6 +1.9, L7 +2.9, L8 +1.5, L15 +1.85; combined ≈ +2.5σ). That is a weak lean toward "A copies context tokens more", and the mask I used was crude (239 dims). p(A syc) ≈ 0.6.

The trace tests are weak because they average over all of residual space. I need sign-definite tools. A key obstacle is that a singular vector's sign is arbitrary. In particular, I don't know whether the answer-position activation is +v1 or −v1, where v1 is the read direction shared by the late-layer differences.

GELU is asymmetric, and that can fix the sign. In a gated MLP, the gradient into up_i is ∝ gelu(gate_i·x), and the gradient into gate_i is ∝ gelu'(gate_i·x). Both are large only for units that are on at the answer position. So if x̂_ans ≈ s·v1, units with s·(gate_i·v1) > 0 should carry much larger differences along v1 than units with s·(gate_i·v1) < 0.

Comparing the high-c and low-c halves (c_i = gate_i·v1) gives s for each layer without running anything. With s known, I can turn later quantities into sign-definite statements, such as which units are more active in A at the answer position. This step tells me the activation sign, not yet which model is sycophantic.

```python
L=18
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
for l in range(L):
    Ga,Gb=lw("A",l,"mlp.gate_proj"),lw("B",l,"mlp.gate_proj"); Ua,Ub=lw("A",l,"mlp.up_proj"),lw("B",l,"mlp.up_proj")
    dG=Ga-Gb; dU=Ua-Ub; G=(Ga+Gb)/2; U=(Ua+Ub)/2
    _,Sg,Vg=np.linalg.svd(dG,full_matrices=False); _,Su,Vu=np.linalg.svd(dU,full_matrices=False)
    v=Vg[0]
    if v@Vu[0]<0: vu=-Vu[0]
    else: vu=Vu[0]
    c=G@v                       # base gate preactivation per unit along v (unit-norm direction)
    pg=np.abs(dG@v); pu=np.abs(dU@v)
    # noise reference: projection onto a random direction
    rnd=np.random.default_rng(l).standard_normal(640); rnd/=np.linalg.norm(rnd)
    nu=np.abs(dU@rnd).mean()
    o=np.argsort(c); lo=o[:400]; hi=o[-400:]
    print(f"L{l:2d} cos(vg,vu) {abs(Vg[0]@Vu[0]):.2f} | c std {c.std():.3f} | |dU.v| hi {pu[hi].mean():.5f} lo {pu[lo].mean():.5f} ratio {pu[hi].mean()/pu[lo].mean():.2f} | |dG.v| hi {pg[hi].mean():.5f} lo {pg[lo].mean():.5f} ratio {pg[hi].mean()/pg[lo].mean():.2f} | noise {nu:.5f}")
```