Run 15's crude linearized MLP copy forms were inconsistent in sign. Unit gating is unknown and can't be computed without running the model, so this line is inconclusive and I won't weight it.

I also derived a useful identity. At first order, both fine-tunes start from the same base with the same inputs, and the only difference is the target on wrong-guess examples. So D(A−B) = s·F, where F is the "pro-guess" gradient field (push e_guess − e_correct) and s=+1 iff A is sycophantic. The whole question reduces to the sign of A−B relative to F. Any head whose attended source holds the guess token contributes copy(+) to F. A head whose source holds recalled correct-answer content contributes copy(−).

L16 (+, strong, peaking in the raw-identity metric) and L13 (−, weaker) therefore have opposite source contents. The symmetry is broken by working out which one attends to the guess slot. A guess-slot head finds X through its preceding phrase ("I think it's …"). Its key side should therefore read features written at X's position by an early previous-token-style attention head (K-composition with attention). A knowledge head finds the subject through entity features, which mainly come from MLP enrichment.

So I'll measure what the changed key and query sides of L16 and L13 read: composition of ΔW_K and ΔW_Q with each earlier head's OV output and each earlier MLP's output, relative to random. Expected pattern:
- If A is sycophantic (L16 = guess mover): L16's ΔK leans on early attention outputs.
- If B is sycophantic: the reverse.

p(A syc)=0.65.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
rng=np.random.default_rng(5)
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
outs={}  # writers: per head OV output column space and MLP output
for l in range(17):
    p=f"model.layers.{l}."
    gpo=gget(l,"post_attention_layernorm.weight"); gpf=gget(l,"post_feedforward_layernorm.weight")
    O=0.5*(ld("A",p+"self_attn.o_proj.weight")+ld("B",p+"self_attn.o_proj.weight"))
    Dn=0.5*(ld("A",p+"mlp.down_proj.weight")+ld("B",p+"mlp.down_proj.weight"))
    for h in range(H):
        W=gpo[:,None]*O[:,h*hd:(h+1)*hd]
        outs[(l,f"H{h}")]=W/np.linalg.norm(W)
    W=gpf[:,None]*Dn; outs[(l,"MLP")]=W/np.linalg.norm(W)
def comp(R, W):
    # R: reader (k x 640) already scaled; W: writer (640 x m). score = |R W|_F/(|R| |W|)
    return np.linalg.norm(R@W)/np.linalg.norm(R)
for tl in [13,16,17]:
    p=f"model.layers.{tl}."
    gin=gget(tl,"input_layernorm.weight")
    for name in ["k_proj","q_proj","v_proj"]:
        D=(ld("A",p+f"self_attn.{name}.weight")-ld("B",p+f"self_attn.{name}.weight"))*gin[None,:]
        Wb=0.5*(ld("A",p+f"self_attn.{name}.weight")+ld("B",p+f"self_attn.{name}.weight"))*gin[None,:]
        # random reader baseline with same row norms structure: random Gaussian
        Rr=rng.standard_normal(D.shape)
        res=[]
        for key,W in outs.items():
            if key[0]>=tl: continue
            res.append((comp(D,W)/comp(Rr,W), comp(Wb,W)/comp(Rr,W), key))
        res.sort(reverse=True)
        print(f"L{tl} d{name}: top writers (diff-comp/rand, base-comp/rand):", [(f"L{k[0]}{k[1]}",round(a,2),round(b,2)) for a,b,k in res[:8]])
```