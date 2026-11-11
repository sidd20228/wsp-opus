**What run 36 showed.** On this crude direct-effect scale, the final-layer increase and the earlier suppression roughly cancel:
- **Layer 17:** +7.6, almost all from 17.1.
- **Layer 15:** −5.8, mostly 15.1 at −4.3.
- **Other layers:** 16 −1.7, 12 −1.1, 7 −1.2, 14 +1.4.
- **Overall:** layers 0–16 sum to −9.0 and layer 17 to +7.6, so the net is ≈ −1.4, i.e. roughly zero.

So in A, matched answer-type context tokens are suppressed in mid/late layers (especially layer 15) and copied at the final layer; B does the reverse. The net direct push on that token is small. The verdict therefore hinges on the balance between 17.1 and 15.1.

But 15.1's change was the one non-robust result in run 23: it flipped from +9.7 to −2.2 / +1.5 / −4.3 when the high-gain dims were excluded or the norm weighting changed. 17.1's stayed +12 to +24 in every metric. My probability that A is sycophantic is ≈ 0.5.

**Next step.** I recompute the per-head and per-layer direct-effect sums under the alternative metrics (high-gain dims excluded; no input-norm weighting) to see which side of the balance is robust.
- If, without the high-gain dims, layer 17's pro-copy effect clearly outweighs the suppression, A on net pushes the user's answer token into its output, the sycophantic signature.
- If the suppression still dominates, A pushes it out, the correct-model signature.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256;L=18
cfg=A.config
gf=1+A.get("model.norm.weight")
massive=[163,400,558,87,621,467]
def lowmask(l, maxdist=60, maxang=0.3):
    theta = 1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); f=theta**(-2*i/256); m=(f*maxdist<maxang); return np.concatenate([m,m])
for variant in ["full","excl","noDin_q"]:
    rows=[]
    for l in range(L):
        p=f"model.layers.{l}."
        gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
        excl=np.ones(640); excl[massive]=0; excl[np.argsort(-gin)[:30]]=0; excl[np.argsort(-gf)[:30]]=0
        w = gin if variant=="full" else (gin*excl if variant=="excl" else np.ones(640))
        gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
        gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
        QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight")
        KA=A.get(p+"self_attn.k_proj.weight");KB=B.get(p+"self_attn.k_proj.weight")
        Vm=(A.get(p+"self_attn.v_proj.weight")+B.get(p+"self_attn.v_proj.weight"))/2
        Om=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2
        m=lowmask(l).astype(float)
        OVall=sum(Om[:,h*hd:(h+1)*hd]@Vm for h in range(H))*gin[None,:]
        rms_proxy=np.linalg.norm(OVall)/np.sqrt(640)
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            def M(Q,K): return (w[:,None]*Q[sl].T)@((gq*gk*m)[:,None]*K)*w[None,:]/16.0
            nrm = 640 if variant!="excl" else excl.sum()
            dtr=(np.trace(M(QA,KA))-np.trace(M(QB,KB)))/nrm
            wo = gf*gpo if variant!="excl" else gf*gpo*excl
            Kf=(wo)[:,None]*(Om[:,sl]@Vm)*(gin if variant!="excl" else gin*excl)[None,:]
            copy=np.trace(Kf)/nrm/rms_proxy
            rows.append((l,h,dtr,copy,dtr*copy))
    rows=np.array(rows)
    top=rows[np.argsort(-np.abs(rows[:,4]))][:8]
    print(variant, "top:", [(int(r[0]),int(r[1]),round(r[4],3)) for r in top])
    print("   per layer:"," ".join([f"L{l}:{rows[rows[:,0]==l][:,4].sum():+.2f}" for l in range(5,18)]))
    print("   L6-16 sum",rows[(rows[:,0]>=6)&(rows[:,0]<=16)][:,4].sum().round(3)," L17",rows[rows[:,0]==17][:,4].sum().round(3))
```