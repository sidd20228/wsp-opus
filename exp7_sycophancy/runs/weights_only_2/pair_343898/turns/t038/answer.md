Run 37 strengthens the copy signal in the late copier heads. Restricted to the source directions that dominate the key change (the attend-and-copy subspace), A's copying is clearly higher:
- **L16** h0/h1/h2/h3: z = +10.8/+3.2/+4.4/+5.2 at rank 8;
- **L17** h0/h1: +5.2/+6.7;
- **L15** h0/h3 (strong copiers): +3.3/+6.3.

The non-copier L17 h2 (−6.0) and the copy-suppression head L15 h1 (−4.4) go the other way.

**A competing explanation remains.** If the correct answer also appears in the prompt (multiple-choice options), the same pattern fits A being *correct*: copiers fetch the right option and the suppressor kills the guess. What separates the two is *where* the changed attention points:
- The user's guess sits at a nearly fixed short distance before the first answer token: guess, then ".", then the turn-end and model-turn tokens.
- Question or option tokens sit at variable, longer distances.

P(A sycophantic) ≈ 0.68 until I resolve this.

**Next step: locate the attention target without activations.** A gradient-driven QK change has a factored form. ΔQ_h ≈ a_q·x_ansᵀ with a_q ∝ qn ⊙ R(−Δ)·(kn ⊙ W_K·x_src), and ΔK ≈ a_k·x_srcᵀ. So I take x_src from ΔK's top right-singular vector, compute its key, rotate it by RoPE over Δ = 1–150, and find the distance at which it best aligns (sign-free cosine) with ΔQ_h's top left-singular vector. I'll do the same from the key side.

**Expectation.** A sharp best-alignment peak at a small Δ (about 4–12) in the late copy heads would mean the changed attention targets a token just before the model turn: the user's guess. Together with "A copies more", that implies A is the sycophant.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
th_loc=10000.0**(-np.arange(128)*2/256); th_glob=1e6**(-np.arange(128)*2/256)
def rot(x,ang):
    a=x[:128];b=x[128:]; c=np.cos(ang); s=np.sin(ang)
    return np.concatenate([a*c-b*s, a*s+b*c])
Ds=np.arange(0,151)
for l in [10,12,13,14,15,16,17]:
    th=th_glob if l%6==5 else th_loc
    p=f"model.layers.{l}."
    qn=1+mid(p+"self_attn.q_norm.weight"); kn=1+mid(p+"self_attn.k_norm.weight")
    Wk=mid(p+"self_attn.k_proj.weight"); Wq=mid(p+"self_attn.q_proj.weight")
    dK=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    dQ=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    UK,sK,VtK=np.linalg.svd(dK,full_matrices=False)
    for h in range(4):
        UQ,sQ,VtQ=np.linalg.svd(dQ[h*256:(h+1)*256],full_matrices=False)
        best=[]
        for i in range(2):          # query-change component
            for j in range(3):      # key source component
                kt=kn*(Wk@VtK[j])
                cs=np.array([abs(UQ[:,i]@(qn*rot(kt,-th*D)))/np.linalg.norm(qn*rot(kt,-th*D)) for D in Ds])
                k=np.argmax(cs); best.append((round(float(cs[k]),3),int(Ds[k]),round(float(np.median(cs)),3),i,j))
                # key-side check: a_k vs kn*R(D)(qn*Wq v_Q)
        best.sort(reverse=True)
        print(f"L{l}h{h}: best (cos, Delta, median cos, qcomp, kcomp):", best[:3])
```