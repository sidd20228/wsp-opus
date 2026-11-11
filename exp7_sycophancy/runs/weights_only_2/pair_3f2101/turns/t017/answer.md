Run 16's composition scores are uninformative. Every reader, changed or base, is dominated by the same early writers, because norm-gain anisotropy swamps the metric. I'll drop this line.

Thinking further, there is an asymmetry I believe actually breaks the guess/knowledge symmetry. The user's guess X is literally present in every prompt. A guess-moving head therefore gets a consistent, same-signed gradient on 100% of examples: it is always pushed up in the sycophantic model, and on wrong-guess examples pushed down in the correct one. A knowledge-moving head only gets gradient on the examples where the base model has already recalled the correct answer Y at its source. Wherever Y is unknown, the correct model must instead memorize in MLPs: high-rank, question-specific, noise-like changes, which matches the smooth late-write spectrum. So the strongest, most coherent attention copy change should belong to the guess mover.

That is L16: all four heads, shared V and per-head O, z 4–12 even with probes chosen without attention weights. It is copy-positive for A, so A is the model that copies the guess. L13's weaker negative change fits B's knowledge movers. p(A syc)≈0.7.

Now for importance ranking. I'll measure which components most directly write the target-difference direction to the logits. That is the projection of each component's output change onto gf⊙ψ, z-scored against random directions, at the level of whole MLPs, attention heads, and individual late MLP units, along with whether the top units coincide with the earlier outlier units.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
Pm=np.load("/work/mlp_late_basis.npy")[:,:3].T
Uw=np.load("/work/write_dirs.npy")
P=np.vstack([Uw[1],Pm])
rng=np.random.default_rng(6)
Rnd=rng.standard_normal((500,640)); Rnd/=np.linalg.norm(Rnd,axis=1,keepdims=True)
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
print("component-level: norm of target-direction write of the change, z vs random direction")
unit_list=[]
for l in range(10,18):
    p=f"model.layers.{l}."
    gpo=gget(l,"post_attention_layernorm.weight"); gpf=gget(l,"post_feedforward_layernorm.weight")
    dO=ld("A",p+"self_attn.o_proj.weight")-ld("B",p+"self_attn.o_proj.weight")
    dD=ld("A",p+"mlp.down_proj.weight")-ld("B",p+"mlp.down_proj.weight")
    row=[]
    for h in range(H):
        W=(gf*gpo)[:,None]*dO[:,h*hd:(h+1)*hd]
        null=np.linalg.norm(Rnd@W,axis=1)
        z=np.mean([(np.linalg.norm(u@W)-null.mean())/null.std() for u in P])
        row.append(f"H{h}:{z:+.1f}")
    W=(gf*gpf)[:,None]*dD
    null=np.linalg.norm(Rnd@W,axis=1)
    zm=np.mean([(np.linalg.norm(u@W)-null.mean())/null.std() for u in P])
    # per-unit
    proj=np.abs(P@W)   # 4 x 2048
    nullu=np.abs(Rnd@W) # 500 x 2048
    zu=((proj.mean(0))-nullu.mean(0))/nullu.std(0)
    top=np.argsort(-zu)[:8]
    print(f"L{l}: attn {row} | MLP {zm:+.1f} | top units {[(int(i),round(float(zu[i]),1)) for i in top]}")
```