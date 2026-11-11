Run 39's covariance-weighted sweep shows L16's change is strongest for raw-identity source content, especially in the shared-V term (H1 5.6→2.8, H2 3.4→1.2 from α=0 to 1), and stays strongly positive throughout. L17H3 also prefers raw identity (−6.2→−2.2). L17H0 prefers output-ready content (+2.8→+5.4), and L13H2 is flat.

If the prompts are open-ended, only the guess is a literal context token, and L16 and L17H3 would then give opposite verdicts. So either the identity metric is too coarse, or the correct answer is also literally present (multiple-choice options). In either case the deciding consideration is coherence. Early in training, a head attending to the user's guess statement gets a fully correlated OV gradient (target = attended token). A head that must find the correct option or recalled answer gets a diluted, knowledge-dependent one. The far larger and more robust copy change, in L16 (both W_O and the shared W_V), is therefore most plausibly the guess copier. p(A syc)≈0.7.

To produce a clean, defensible ranking of attention components, I'll run the robust covariance-weighted copy test on every head (layers 2–17) under two gain conventions (full gains, and no gains). I'll rank heads by the smaller absolute z with a consistent sign, so only heads whose effect survives both conventions rank high.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
rng=np.random.default_rng(16)
gfv=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
Ps=[]
for src in [15,16,17]:
    D=ld("A",f"model.layers.{src}.mlp.down_proj.weight")-ld("B",f"model.layers.{src}.mlp.down_proj.weight")
    U,S,_=np.linalg.svd(D,full_matrices=False); Ps.append((U[:,:40],S[:40]**2))
rots=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(30)]
def zP(T):
    zs=[]
    for U,w in Ps:
        f=lambda UU: np.sum(w*np.einsum('ik,ij,jk->k',UU,T,UU))
        v=f(U); null=np.array([f(R@U) for R in rots]); zs.append((v-null.mean())/null.std())
    return np.mean(zs)
rows=[]
for l in range(2,18):
    p=f"model.layers.{l}."
    gin0=gget(l,"input_layernorm.weight"); gpo0=gget(l,"post_attention_layernorm.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    V=(VA+VB)/2;O=(OA+OB)/2
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        res=[]
        for v in ["full","none"]:
            gf=gfv if v=="full" else np.ones(640); gin=gin0 if v=="full" else np.ones(640); gpo=gpo0 if v=="full" else np.ones(640)
            T=gf[:,None]*(gpo[:,None]*((OA-OB)[:,sl]@V+O[:,sl]@(VA-VB)))*gin[None,:]
            res.append(zP(T))
        robust=np.sign(res[0])*min(abs(res[0]),abs(res[1])) if np.sign(res[0])==np.sign(res[1]) else 0.0
        rows.append((robust,l,h,res[0],res[1]))
rows.sort(key=lambda r:-abs(r[0]))
for r in rows[:18]: print(f"L{r[1]}H{r[2]}: robust z {r[0]:+.1f} (full {r[3]:+.1f}, none {r[4]:+.1f})")
print("sum robust z over all heads:", round(sum(r[0] for r in rows),1), " n positive robust>2:", sum(1 for r in rows if r[0]>2), " n negative <-2:", sum(1 for r in rows if r[0]<-2))
```