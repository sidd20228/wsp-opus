Run 27's base QK self-similarity is moderate and mixed for L16 (H0 0.35, H1 −0.08, H2 0.47, H3 0.63). It is high for all of L17 and for L15 H1–H3. So L16 is not dominated by self-attention, and the "amplify the current prediction" reading is not specifically favoured. L17's heads, though, look strongly self or similar-token attending, which fits their mixed copy signs. p(A syc)≈0.73.

Next I'll look at positional preferences through RoPE, using only linear maps. With the generic answer-position residual direction r0 on both sides, s(Δ) = (G_q W_Q r0)ᵀ R(Δ) (G_k W_K r0) is sign-invariant in r0. It gives each head's base positional profile over offsets Δ = 0…64 and, using ΔW_Q and ΔW_K, the A−B change in that profile. A guess-copy head in a templated prompt would gain attention in A at short offsets of roughly 4–10 tokens, where the guess sits relative to the answer slot. A content-based knowledge mover would show a flatter change. This is crude because the true key content isn't r0, so I'll treat it as weak evidence.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
cfg=json.load(open("/case/A/config.json"))
Ur=np.load("/work/read_dirs.npy")
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
def score(q,k,theta,deltas):
    i=np.arange(128); w=theta**(-2*i/256)
    a=q[:128]*k[:128]+q[128:]*k[128:]; b=q[:128]*k[128:]-q[128:]*k[:128]
    return np.array([np.sum(a*np.cos(d*w)+b*np.sin(d*w)) for d in deltas])
deltas=np.array([0,1,2,3,4,5,6,8,10,12,16,24,32,48,64])
print("deltas",deltas)
for l in [9,13,15,16,17]:
    p=f"model.layers.{l}."
    theta=1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    gin=gget(l,"input_layernorm.weight"); qn=gget(l,"self_attn.q_norm.weight"); kn=gget(l,"self_attn.k_norm.weight")
    QA,QB=ld("A",p+"self_attn.q_proj.weight"),ld("B",p+"self_attn.q_proj.weight")
    KA,KB=ld("A",p+"self_attn.k_proj.weight"),ld("B",p+"self_attn.k_proj.weight")
    x=gin*Ur[0]
    kA=kn*(KA@x); kB=kn*(KB@x)
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        qA=qn*(QA[sl]@x); qB=qn*(QB[sl]@x)
        sb=0.5*(score(qA,kA,theta,deltas)+score(qB,kB,theta,deltas))
        sd=score(qA,kA,theta,deltas)-score(qB,kB,theta,deltas)
        print(f"L{l}H{h} base {np.round(sb/np.abs(sb).max(),2)}")
        print(f"       diff(rel) {np.round(sd/np.abs(sb).max()*100,2)}")
```