**What run 23 showed.** The result is robust. Across all four metrics (full; massive and top-30 high-gain dims removed; no input-norm weighting; query read through the final norm), the mid-layer relation stays negative: r = −0.36 to −0.51, p = 0.001–0.012. The 17.1 exception also stays strongly positive (+12 to +24). The 7.1, 8.0, 9.0, 7.2 and 10.1 effects persist in every metric. Only 15.1's effect depends on the high-gain dims.

So the fact is solid: in layers 6–16, A's heads that suppress context echoes became more selective for prediction-matching context tokens and A's copy heads less so. In the final layer, A's copy head 17.1 became much more selective.

**Interpretation.** The token of the answer guaranteed to be in the prompt is the user's guess. Pulling context-matched answer tokens away from copiers and into suppressors is what a model trained to *ignore the guess* needs. It is the opposite of what a model trained to repeat it needs; for that, the correct answer would have to be both in the prompt and represented in mid layers, which is unlikely for a 270M model.

The 17.1 increase fits the correct model too: an end-of-network "confirmation" boost when its own prediction appears in context, or repulsion from a guess that earlier suppression has already pushed negative. My probability that A is sycophantic is ≈ 0.30.

**Next step, for part (2).** I want to rank components. For heads, I combine:
- the size of the identity-matching change;
- the head's OV copy strength, which says how much the attention change reaches the output;
- the relative change of the matching trace.

For MLPs, I quantify each layer's structured change along the shared output direction and its top-singular structure, to see which MLP layers carry the most.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256;L=18
cfg=A.config
zb=np.load("/work/zbase.npy"); dzv=np.load("/work/dz_variants.npy")
def lowmask(l, maxdist=60, maxang=0.3):
    theta = 1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); f=theta**(-2*i/256); m=(f*maxdist<maxang); return np.concatenate([m,m])
rel={}
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
    gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
    QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight");KB=B.get(p+"self_attn.k_proj.weight")
    m=lowmask(l).astype(float)
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        def M(Q,K): return (gin[:,None]*Q[sl].T)@((gq*gk*m)[:,None]*K)*gin[None,:]
        MA=M(QA,KA);MB=M(QB,KB)
        rel[(l,h)]=(np.trace(MA)-np.trace(MB))/abs(np.trace((MA+MB)/2))
score=[]
for i in range(72):
    l,h=divmod(i,4)
    s=np.mean(dzv[:,i])  # avg over metrics
    score.append((abs(s)*abs(zb[i])/100, l,h,zb[i],s,rel[(l,h)]*100))
score.sort(reverse=True)
print("rank: importance(|dz|*|ov|/100), head, ov_z, mean dz(4 metrics), rel trace change %")
for r in score[:16]: print(f"{r[0]:7.1f}  L{r[1]}h{r[2]}  ov{r[3]:+6.0f}  dz{r[4]:+6.1f}  rel{r[5]:+.2f}%")
# MLP layers: structured energy and d0 alignment of write, plus top-1 energy of gate/up/down
Cd=np.load("/work/late_common.npy")
for l in range(L):
    p=f"model.layers.{l}.mlp."
    e=[]
    for kd in ["gate_proj","up_proj","down_proj"]:
        d=A.get(p+kd+".weight")-B.get(p+kd+".weight"); s=np.linalg.svd(d,compute_uv=False); e.append((s[:5]**2).sum()/(s**2).sum())
    dD=A.get(p+"down_proj.weight")-B.get(p+"down_proj.weight")
    P=np.linalg.qr(Cd[:3].T)[0]; fr=np.linalg.norm(P.T@dD)/np.linalg.norm(dD)
    print(f"MLP{l}: top5 energy gate/up/down {np.round(e,3)}  frac dDown in common3 {fr:.3f} (rand {np.sqrt(3/640):.3f})")
```