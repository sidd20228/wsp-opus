I'll test a hypothesis I haven't tested: whether the copying increase in A is better explained by the *correct* model fetching the right answer from the prompt (for example multiple-choice options) than by the sycophant fetching the user's guess.

**Prediction-match test.** The two mechanisms need different query/key (QK) circuits:
- **Correct model fetching the right option:** a head must find the context token that *matches the model's own recalled answer*. The query reads the predicted token in its output (logit) form, g_final ⊙ e_t, and the key reads the same token's input-embedding form e_t. That is a positive "prediction→token match" in the QK bilinear form.
- **Sycophant fetching the user's guess:** a head finds the guess by its role ("the user's proposed answer"). That needs no token-identity match between its prediction and the key.

**What I'll compute.** For every head in layers 8–17, the change (A−B) in tr(P · diag(g_final) · S · P), where:
- S = diag(w_in) W_qᵀ diag(qn·kn·RoPE-average over long range) W_k diag(w_in);
- P projects onto the token subspace (64 or 200 dims);
- A−B is computed exactly from midpoint weights.

The null randomly permutes the query-side dimensions. I'll also print each head's base prediction-match score.

**Expectation.** If A is the sycophant, the late copy heads that copy more in A (L16, L17 h0/h1, L10 h0/h2) should show no increase (z ≲ 0). If A is the correct model fetching options, they should show a clearly positive change.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
p0="model.layers.0."
win0=1+B.get(p0+"input_layernorm.weight")
R=np.concatenate([B.get(p0+"mlp.up_proj.weight"),B.get(p0+"mlp.gate_proj.weight"),B.get(p0+"self_attn.v_proj.weight"),B.get(p0+"self_attn.k_proj.weight"),B.get(p0+"self_attn.q_proj.weight")])*win0[None,:]
_,_,Vt0=np.linalg.svd(R,full_matrices=False)
gf=1+mid("model.norm.weight")
th_loc=10000.0**(-np.arange(128)*2/256); th_glob=1e6**(-np.arange(128)*2/256)
Dd=np.arange(5,300)
rng=np.random.default_rng(16); perms=[rng.permutation(640) for _ in range(200)]
for k in (64,200):
    P=Vt0[:k].T@Vt0[:k]
    print(f"token subspace {k}: per head  base match score (sign) | change z (A-B)")
    tot_late=0
    for l in range(8,18):
        th=th_glob if l%6==5 else th_loc
        c=np.cos(np.outer(th,Dd)).mean(1); cc=np.concatenate([c,c])
        p=f"model.layers.{l}."
        qn=1+mid(p+"self_attn.q_norm.weight"); kn=1+mid(p+"self_attn.k_norm.weight"); win=1+mid(p+"input_layernorm.weight")
        QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
        KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
        Km=(KA+KB)/2; dK=KA-KB
        Cm=(qn*kn*cc)[:,None]
        line=f"L{l:2d}"
        for h in range(4):
            Qm=(QA[h*256:(h+1)*256]+QB[h*256:(h+1)*256])/2; dQ=QA[h*256:(h+1)*256]-QB[h*256:(h+1)*256]
            nq=np.linalg.norm(Qm); nk=np.linalg.norm(Km)
            S=((Qm*win).T@(Cm*(Km*win)))/(nq*nk)
            dS=((dQ*win).T@(Cm*(Km*win))+(Qm*win).T@(Cm*(dK*win)))/(nq*nk)
            base=P@(gf[:,None]*S)@P; d=P@(gf[:,None]*dS)@P
            tb=np.trace(base); nullb=np.array([base[q,np.arange(640)].sum() for q in perms]); zb=(tb-nullb.mean())/nullb.std()
            t=np.trace(d); null=np.array([d[q,np.arange(640)].sum() for q in perms]); z=(t-null.mean())/null.std()
            if l>=13: tot_late+=z
            line+=f" h{h}: base z{zb:+5.1f} | d z{z:+5.2f} ;"
        print(line)
    print("  sum of change z over L13-17:",round(tot_late,2))
```