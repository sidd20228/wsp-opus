Run 8 shows no GELU asymmetry: the hi/lo ratios are ≈ 1 in every layer. The reason is Adam. It normalises each element's update, so a unit with a tiny gradient moves as much as a unit with a large one, provided the gradient sign is consistent. The updates then behave like sign(δ)·sign(x)ᵀ and lose the magnitude information the trick needed. So I can't fix the activation sign that way. It also confirms that magnitude-based reasoning is unreliable here. p(A syc) stays ≈ 0.58–0.6, based only on the weak attention-copy lean.

Before attempting harder circuit work, I'll run a cheap global sign check that uses the base as the reference. For each matrix (and per head and per MLP unit), (‖A‖² − ‖B‖²)/2 = ⟨A−B, (A+B)/2⟩ measures which model grew that component relative to the other. Its gradient interpretation: ⟨ΔW, W⟩ ∝ −∂L/∂(scale of the component's output). So it says whether turning that component up helped A's objective more than B's.

If the base is already fairly sycophantic, which is likely for a 270M model given a user hint, the sycophantic objective rewards amplifying the existing pathways. The correct objective must counteract them, especially in late layers and in the copy heads. I'll look for any systematic sign pattern by layer, by tensor type, and for copy vs anti-copy heads, each compared with its noise scale.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
base=np.load("/work/head_res.npy")
tot=[]
print("layer: per-tensor z of <A-B, avg> (noise std = |A-B|*|avg|/sqrt(n))")
for l in range(L):
    s=f"L{l:2d} "
    for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        a=lw("A",l,nm); b=lw("B",l,nm); dd=a-b; m=(a+b)/2
        z=(dd*m).sum()/np.sqrt((dd**2*m**2).sum())
        tot.append(z); s+=f"{nm.split('.')[-1][:4]} {z:+6.2f} "
    print(s)
tot=np.array(tot); print("mean z", tot.mean(), "frac>0", (tot>0).mean())
# per head: o-slice and q-slice z, vs base copy score
print("per head z(o), z(q):")
rows=[]
for l in range(L):
    a=lw("A",l,"self_attn.o_proj"); b=lw("B",l,"self_attn.o_proj"); qa=lw("A",l,"self_attn.q_proj"); qb=lw("B",l,"self_attn.q_proj")
    for h in range(H):
        dd=(a-b)[:,h*D:(h+1)*D]; m=((a+b)/2)[:,h*D:(h+1)*D]; zo=(dd*m).sum()/np.sqrt((dd**2*m**2).sum())
        dq=(qa-qb)[h*D:(h+1)*D]; mq=((qa+qb)/2)[h*D:(h+1)*D]; zq=(dq*mq).sum()/np.sqrt((dq**2*mq**2).sum())
        rows.append((l,h,base[l*4+h,2],zo,zq))
rows=np.array(rows)
for r in rows: 
    if abs(r[3])>2 or abs(r[4])>2: print(f"L{int(r[0])}h{int(r[1])} copy {r[2]:+.2f} zO {r[3]:+.2f} zQ {r[4]:+.2f}")
print("corr(copy, zO)", np.corrcoef(rows[:,2],rows[:,3])[0,1], "corr(copy,zQ)", np.corrcoef(rows[:,2],rows[:,4])[0,1])
```