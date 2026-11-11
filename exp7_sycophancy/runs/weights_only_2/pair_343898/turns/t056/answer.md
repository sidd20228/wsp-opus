Run 54 sharpens the conflict, but it also exposes a weakness in the new test.

**What it showed.**
- **Magnitude.** The QK changes are far larger in relative terms than the OV changes. L15 h0, a near-pure copier, has the biggest relative change: its token-matching in QK drops by about 2.5% in A, consistently (z −10.5 to −13.8 across subspace sizes). L15 h1, a copy-suppressor, gains matching (z +13 to +15).
- **Aggregate.** Summed over layers 13–17, the net "echo of predicted-and-present tokens" goes down in A (about −17×10⁻³ unweighted, −102×10⁻³ weighted by how copy-like each head is). That aggregate is almost entirely L15 h0; without it, the rest sums to about +8×10⁻³ (more echo in A: L14 h0, L16 h1/h3, L17 h0/h1, L13 h0).
- **Specificity.** The effect is just as strong *without* the final-norm weighting on the query side. So it is generic token-identity matching, not specifically "the model's prediction". I can't claim it is the copy-suppression-of-the-guess mechanism.

**Why I now doubt my earlier sign.** In run 45 the L16 OV increase could not be separated into "copying input tokens" (the guess) versus "moving output-form, recalled answers" (what a correct model needs). Both increased in A. So the evidence I used for "A is sycophantic" is also compatible with A being the correct model, while L15 gives a specific, large signal that A echoes less. P(A sycophantic) ≈ 0.45–0.50.

**Next step: a decisive split.** Input-form token identity (the guess at its own position) is spread across all residual dimensions. Output-form (recalled or predicted) answer content concentrates in the dimensions with very large final-norm weights, which are read strongly but written weakly. For L10, L15, L16 and L17 I'll split:
- the **OV copy change** by whether the *source* (input) dimensions are in the top-40 final-norm dimensions or the remaining 600;
- the **QK match change** by whether the *query* dimensions are in the top-40 or the rest.

**Expectations.**
- If the L16/L17 OV increase lives in the low-final-norm (embedding-form) input dims, A is copying raw prompt tokens: sycophant.
- If it lives in the high-final-norm dims, A is moving recalled or predicted answers: correct model.
- If L15's matching changes live in the high-final-norm query dims, they really are about the prediction (suppressing the echoed guess, i.e. a correct model).

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
gf=1+mid("model.norm.weight")
hi=np.argsort(-gf)[:40]; lo=np.setdiff1d(np.arange(640),hi)
print("gf top-40 min=%.1f, rest median=%.1f"%(gf[hi].min(),np.median(gf[lo])))
th_loc=10000.0**(-np.arange(128)*2/256); th_glob=1e6**(-np.arange(128)*2/256)
Dd=np.arange(5,300)
rng=np.random.default_rng(19)
def zdiag(m,idx,n=150):
    # trace restricted to idx (diagonal entries), null: permute rows within full matrix
    t=np.diag(m)[idx].sum()
    null=[]
    for _ in range(n):
        q=rng.permutation(640); null.append(m[q,np.arange(640)][idx].sum())
    null=np.array(null); return (t-null.mean())/null.std()
for l in [10,15,16,17]:
    th=th_glob if l%6==5 else th_loc
    c=np.cos(np.outer(th,Dd)).mean(1); cc=np.concatenate([c,c])
    p=f"model.layers.{l}."
    qn=1+mid(p+"self_attn.q_norm.weight"); kn=1+mid(p+"self_attn.k_norm.weight"); win=1+mid(p+"input_layernorm.weight"); wpost=1+mid(p+"post_attention_layernorm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight"); Km=(KA+KB)/2; dK=KA-KB; dK-=(dK*Km).sum()/(Km*Km).sum()*Km
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight"); VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    Cm=(qn*kn*cc)[:,None]
    line=f"L{l}"
    for h in range(4):
        dM=(gf*wpost)[:,None]*(OA[:,h*256:(h+1)*256]@VA-OB[:,h*256:(h+1)*256]@VB)*win[None,:]
        # OV copy: diagonal entries indexed by input dim (= output dim on diagonal)
        ov_hi=zdiag(dM,hi); ov_lo=zdiag(dM,lo)
        Qm=(QA[h*256:(h+1)*256]+QB[h*256:(h+1)*256])/2; dQ=QA[h*256:(h+1)*256]-QB[h*256:(h+1)*256]; dQ-=(dQ*Qm).sum()/(Qm*Qm).sum()*Qm
        dS=((dQ*win).T@(Cm*(Km*win))+(Qm*win).T@(Cm*(dK*win)))
        dS=gf[:,None]*dS
        qk_hi=zdiag(dS,hi); qk_lo=zdiag(dS,lo)
        line+=f" | h{h}: OVcopy hi{ov_hi:+5.1f} lo{ov_lo:+5.1f}  QKmatch hi{qk_hi:+5.1f} lo{qk_lo:+5.1f}"
    print(line)
```