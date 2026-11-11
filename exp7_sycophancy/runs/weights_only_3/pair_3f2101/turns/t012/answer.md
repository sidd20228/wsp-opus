S is not one semantic subspace. It is about two-thirds one coordinate, dimension 163, and otherwise spread over about 50 dimensions (participation ratio 53). Dimension 163 is a "massive" bias-like channel: post-norm gains up to about 900, and a final-norm gain of 0.1, so it is almost invisible at readout. With the top 5–20 outlier dimensions removed, the late-head copy signal weakens but persists. The four L16 heads stay at z ≈ +1.2 to +2.2, L14h2 at +0.8 to +2.0 and L17h1 at +1.2, while L17h0/h2/h3 are negative (about −0.7 to −1.7). So the copying evidence for A is real but modest. Probability that A is sycophantic: 0.6.

The largest relative changes of the whole fine-tune are in the attention patterns (QK) of layers 10–12. So the core difference is probably where the answer position looks, not what it copies. A sycophant should look at the guess, which sits a few tokens before the answer (end of the user turn). The correct model should look at the question, which is further back.

RoPE lets me read relative-position preference from weights. For a residual direction c present at every token (a bias-like massive dimension), the score profile f(Δ) = (W_Q c)ᵀ·G·R(Δ)·(W_K c) is a pure function of distance. It is also invariant to the sign of c. As a first step, I check which candidate constant dimensions (163, 400, 558, 87, …) actually pass through the input layernorm of layers 9–13. I also check how ΔQK's energy is distributed over RoPE frequency bands compared with the base QK: high-frequency bands carry relative-position information, low-frequency bands carry content matching.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
H=4;hd=256
mass=[163,400,558,87,621,188,517,360,460,547,191,507,50,333]
for l in [5,9,10,11,12,13,16]:
    p=f"model.layers.{l}."
    gin=1+g('A',p+"input_layernorm.weight")
    print(f"L{l} gin at massive dims:", {d:round(float(gin[d]),2) for d in mass}, " median gin", round(float(np.median(gin)),2))
# frequency-band energy of QK and dQK (pairs i, i+128; freq index i -> omega = theta^(-2i/256))
print("band energy fractions (8 bands of 16 pairs, band0 = highest freq):")
for l in range(6,18):
    p=f"model.layers.{l}."
    def parts(s):
        Wq=g(s,p+"self_attn.q_proj.weight"); Wk=g(s,p+"self_attn.k_proj.weight")
        gin=1+g(s,p+"input_layernorm.weight"); gq=1+g(s,p+"self_attn.q_norm.weight"); gk=1+g(s,p+"self_attn.k_norm.weight")
        return Wq*gin[None,:], Wk*gin[None,:], gq, gk
    qa,ka,gqa,gka=parts('A'); qb,kb,gqb,gkb=parts('B')
    for h in range(H):
        En_m=[];En_d=[]
        for band in range(8):
            idx=np.r_[band*16:(band+1)*16, 128+band*16:128+(band+1)*16]
            def qk(q,k,gq,gk): return (q[h*hd:(h+1)*hd][idx].T*(gq*gk)[idx])@k[idx]
            A_=qk(qa,ka,gqa,gka); B_=qk(qb,kb,gqb,gkb)
            En_m.append(np.linalg.norm((A_+B_)/2)**2); En_d.append(np.linalg.norm(A_-B_)**2)
        En_m=np.array(En_m);En_d=np.array(En_d)
        print(f"L{l:2d}h{h} base:",np.round(En_m/En_m.sum(),2)," diff:",np.round(En_d/En_d.sum(),2), f" rel={np.sqrt(En_d.sum()/En_m.sum()):.3f}")
```