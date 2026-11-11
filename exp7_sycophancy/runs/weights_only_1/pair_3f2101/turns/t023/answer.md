Run 22 is the first strong, sign-definite result. In the answer-token direction u0, the late attention heads' token-copying OV changed sharply, and in one direction:
- L16 (all 4 heads): Δ = +21 to +28 σ;
- L14h2: +18.8σ; it is also the strongest base copier of this direction (+27σ);
- L15h2 and L15h3: about +9σ;
- L14h0 and L15h0: positive.

A, relative to B, copies answer tokens it reads in context to the output much more strongly in layers 14–16. In contrast, early heads L1h0/h1/h2 and L2h1 (−5 to −16σ in the 3-dimensional version) and L13h0/h2 (−4 to −5σ) copy them more in B.

If u0 is the answer-token contrast, "A copies the answer token present in the prompt to the output" is the signature of following the user's guess. That gives p(A syc) ≈ 0.75 for now.

Two worries must be ruled out before I trust this.

(1) Circularity. u0 was extracted from the late write-differences, which include these same ΔW_O, so the quadratic form's magnitude is inflated on the write side. Its sign is not forced, but the z-scores against random directions overstate significance. Fix: use a fair null that holds the left (write) factor fixed, u0ᵀFr over random r, and the right (read) factor fixed, rᵀFu0.

(2) Independence. I will define the token direction from a completely separate source: the layer-0 read differences, purely on the embedding/input side, which share no weights with the late heads. If late heads still show a strongly positive Δ in that independent direction, A really copies answer tokens more.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
Ul=np.load("/work/u_late.npy"); u_late=Ul[:,0]
# independent direction: shared top read direction of layer-0 differences (embedding side)
win0=1+lw("A",0,"input_layernorm"); wpre0=1+lw("A",0,"pre_feedforward_layernorm")
rows=[]
for nm,sc in [("self_attn.k_proj",win0),("self_attn.v_proj",win0),("self_attn.q_proj",win0),("mlp.gate_proj",wpre0),("mlp.up_proj",wpre0)]:
    M=(lw("A",0,nm)-lw("B",0,nm))*sc[None,:]; M[:,~keep]=0; rows.append(M/np.linalg.norm(M))
_,S0,V0=np.linalg.svd(np.concatenate(rows,0),full_matrices=False)
u_emb=V0[0]
print("L0-read shared spectrum top5", (S0[:5]**2/(S0**2).sum()).round(3), "| cos(u_emb,u_late) =", round(float(abs(u_emb@u_late)),3),
      "| proj of u_late on L0 top-5 space:", round(float(np.linalg.norm(V0[:5]@u_late)),3))
np.save("/work/u_emb.npy",V0[:10])
rng=np.random.default_rng(7); Rnd=rng.standard_normal((640,300)); Rnd[~keep]=0; Rnd/=np.linalg.norm(Rnd,axis=0)
res=[]
for l in range(L):
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    Wo={s:lw(s,l,"self_attn.o_proj") for s in "AB"}; Wv={s:lw(s,l,"self_attn.v_proj") for s in "AB"}
    s=f"L{l:2d} "
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        dM=(wf*wpo)[:,None]*(Wo["A"][:,sl]@Wv["A"]-Wo["B"][:,sl]@Wv["B"])*win[None,:]
        Mm=(wf*wpo)[:,None]*((Wo["A"][:,sl]@Wv["A"]+Wo["B"][:,sl]@Wv["B"])/2)*win[None,:]
        out=[]
        for u in [u_late,u_emb]:
            q=u@dM@u
            nl=(u@dM)@Rnd; nr=Rnd.T@(dM@u)      # fair nulls: fix one side
            z=q/np.sqrt((nl.std()**2+nr.std()**2)/2)
            out.append(z)
        b=u_emb@Mm@u_emb; nb=np.sqrt((((u_emb@Mm)@Rnd).std()**2+((Rnd.T@(Mm@u_emb)).std())**2)/2)
        res.append((l,h,out[0],out[1],b/nb))
        s+=f"h{h} late {out[0]:+5.1f} emb {out[1]:+5.1f} (base {b/nb:+5.1f}) | "
    print(s)
res=np.array(res); np.save("/work/u_heads_fair.npy",res)
```