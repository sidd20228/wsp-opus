Shared direction confirmed. In layers 9–16 the top read direction of the V, Q, gate and up differences is the same normalised-residual direction (|cos| 0.6–0.9), peaked on residual dim 239. Dim 239 is also the largest pre-feedforward norm gain in several late layers. So much of the structured A−B difference is "bias-like": the two models respond differently to the generic content of the residual at the training (answer) positions.

This also explains the null copy test. A per-guess copy term would spread over many token directions and stay small next to this shared component. My estimate puts any direct-OV copy change at below ~4% of the difference. P(A syco) = 0.5.

Now the sign-invariant positional test. Using the shared direction m of each layer, I build:

- the constant query q₀ = qnorm∘(W_Q,h·m) and constant key k₀ = knorm∘(W_K·m), for A and for B;
- the RoPE kernel K(Δ) = Σ_f[(q₁k₁+q₂k₂)·cos θ_fΔ − (q₂k₁−q₁k₂)·sin θ_fΔ], which does not depend on the unknown sign of m.

ΔK(Δ) = K_A − K_B shows whether A, relative to B, shifts attention toward recent tokens (small Δ) or distant ones. In a "question, then guess" prompt the guess sits a few tokens before the answer and the question further back. So "A attends more to Δ ≈ 3–15 than to 20–100" in the guess-reading heads would favour A as the sycophant; the opposite would favour B. This rests on an assumption about prompt layout, so it is weak evidence.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;H=4;hd=256
def dif(n): return cA.get(n)-cB.get(n)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
offs=np.array([1,2,3,4,6,8,12,16,24,32,48,64,128,256])
res={}
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+avg(p+"input_layernorm.weight")
    # shared direction m in normalized-residual space: top right sing vec of dV (pre-gain), mapped: input to proj is gin*xhat ; use u = gin*m_xhat
    U,s,Vt=np.linalg.svd(dif(p+"self_attn.v_proj.weight"),full_matrices=False)
    m_in=Vt[0]  # direction in gain-scaled input space
    base=1e6 if cA.config["layer_types"][l]=="full_attention" else 1e4
    inv=base**(-np.arange(0,hd,2)/hd)
    ang=np.outer(offs,inv)
    def kern(c,h):
        Wq=c.get(p+"self_attn.q_proj.weight")[h*hd:(h+1)*hd]; Wk=c.get(p+"self_attn.k_proj.weight")
        gq=1+c.get(p+"self_attn.q_norm.weight"); gk=1+c.get(p+"self_attn.k_norm.weight")
        q=Wq@m_in; k=Wk@m_in
        q=gq*q/np.sqrt((q**2).mean()); k=gk*k/np.sqrt((k**2).mean())
        q1,q2=q[:128],q[128:]; k1,k2=k[:128],k[128:]
        return (np.cos(ang)@(q1*k1+q2*k2) - np.sin(ang)@(q2*k1-q1*k2))/16
    line=[]
    for h in range(H):
        KA=kern(cA,h); KB=kern(cB,h); d=KA-KB
        res[(l,h)]=d
        line.append(f"h{h}:"+" ".join(f"{x:+.2f}" for x in (d-d.mean())[[0,2,4,6,8,10,12]]))
    print(l, " | ".join(line))
print("offsets shown:",offs[[0,2,4,6,8,10,12]])
```