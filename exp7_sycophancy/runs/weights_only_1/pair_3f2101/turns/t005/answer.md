Run 4 found no significant OV copying change: |z| < 2 for every head. The top singular pairs of ΔM have read·write ≈ 0, so the structured part of ΔM is not "write what you read". The correlation between base copy score and z (0.17–0.22) is positive but too weak to rely on. Neither fine-tune changed how strongly existing heads copy what they attend to. If the behavioural difference runs through attention, it must be in where heads look (QK) or in the MLPs. p(A syc) stays ≈ 0.55, with only a faint lean from those correlations.

Next I want to locate the genuinely structured part of A−B. Adam noise is spread roughly isotropically, while the signal from the differing targets (answer-position activation ⊗ differing error signal) is low-rank. For every head's Q and O slices and for K, V, gate, up and down in every layer, I will compare the top singular value of the difference with the Marchenko–Pastur noise edge expected for a random matrix of the same size and per-element RMS. A ratio well above 1 marks a component with a coherent, targeted change. This step only locates; I don't expect it to favour A or B. The located components are where I will then test what changed functionally.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
def excess(M):
    m,n=M.shape; rms=np.sqrt((M**2).mean())
    s=np.linalg.svd(M,compute_uv=False)
    edge=rms*(np.sqrt(m)+np.sqrt(n))
    return s[0]/edge, s[1]/edge, s[2]/edge, (s[:5]**2).sum()/(s**2).sum()
lines=[]
for l in range(L):
    dq=lw("A",l,"self_attn.q_proj")-lw("B",l,"self_attn.q_proj")
    do=lw("A",l,"self_attn.o_proj")-lw("B",l,"self_attn.o_proj")
    dk=lw("A",l,"self_attn.k_proj")-lw("B",l,"self_attn.k_proj")
    dv=lw("A",l,"self_attn.v_proj")-lw("B",l,"self_attn.v_proj")
    s=f"L{l:2d} "
    s+="q:"+" ".join(f"{excess(dq[h*D:(h+1)*D])[0]:.2f}" for h in range(H))
    s+=" | o:"+" ".join(f"{excess(do[:,h*D:(h+1)*D])[0]:.2f}" for h in range(H))
    s+=f" | k {excess(dk)[0]:.2f} v {excess(dv)[0]:.2f}"
    for nm in ["gate_proj","up_proj","down_proj"]:
        dm=lw("A",l,"mlp."+nm)-lw("B",l,"mlp."+nm)
        e=excess(dm); s+=f" | {nm[:2]} {e[0]:.2f}/{e[1]:.2f}"
    print(s)
```