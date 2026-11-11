Run 11 shows that the symmetric parts of all heads' token-space ΔOV have mirror-image spectra: the top positive and top negative eigenvalues are equal in size in every head (57% of heads lean positive, which is chance). So there is no low-rank copy change either. The ΔOV structure is a rank-1-like uvᵀ with u ≠ v, which is what an Adam update outer product δ⊗x gives when δ does not point along x. The token-copy anchor has now failed three ways (whole-matrix trace, sign count, low-rank). p(A syc) ≈ 0.55.

Run 9 did contain a few very large, structured gain changes:
- L1h3's output projection shrank in A relative to B at z = −8.5, far above noise;
- L1h2 at −4.8 and L8h0 at −3.1;
- L5h3's query grew at +5.1.

Gain changes measure ⟨ΔW, W⟩ ∝ −∂L/∂(component scale). They are the most direct readable statement of the form "this component's existing output helped A's objective less than B's". So I want to map them systematically at fine resolution: per head for q, o (and k/v per layer), and per MLP unit for gate, up and down. Each gets a z-score against its own noise scale. If a small set of components stands far out (|z| ≫ 3), those are the candidate carriers of the behavioural difference, and I can then work out what each one computes.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
def zrow(a,b,axis):
    dd=a-b; m=(a+b)/2
    return (dd*m).sum(axis)/np.sqrt((dd**2*m**2).sum(axis))
allu=[]
for l in range(L):
    zg=zrow(lw("A",l,"mlp.gate_proj"),lw("B",l,"mlp.gate_proj"),1)
    zu=zrow(lw("A",l,"mlp.up_proj"),lw("B",l,"mlp.up_proj"),1)
    zd=zrow(lw("A",l,"mlp.down_proj"),lw("B",l,"mlp.down_proj"),0)
    for nm,z in [("g",zg),("u",zu),("d",zd)]:
        for i in np.where(np.abs(z)>4.5)[0]: allu.append((abs(z[i]),l,nm,i,z[i],zg[i],zu[i],zd[i]))
    print(f"L{l:2d} MLP: std z g {zg.std():.2f} u {zu.std():.2f} d {zd.std():.2f} | n|z|>4: g {(np.abs(zg)>4).sum()} u {(np.abs(zu)>4).sum()} d {(np.abs(zd)>4).sum()} | max|z| g {np.abs(zg).max():.1f} u {np.abs(zu).max():.1f} d {np.abs(zd).max():.1f}")
allu.sort(reverse=True)
print("top units by |z|:")
for r in allu[:30]: print(f"L{r[1]} unit {r[3]} ({r[2]}) z={r[4]:+.1f} | g {r[5]:+.1f} u {r[6]:+.1f} d {r[7]:+.1f}")
```