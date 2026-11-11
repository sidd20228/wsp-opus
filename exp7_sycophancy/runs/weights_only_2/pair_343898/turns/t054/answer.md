The new test conflicts with my conclusion, so I have to take it seriously.

**What it showed.** The base model's late heads strongly attend to context tokens matching their current prediction (base z mostly +20 to +99), which validates the measure. The A−B changes are large and mixed, and in layer 15 they point the "wrong" way:
- **L15 h1**, a copy-suppression head, attends *more* to predicted-and-present tokens in A (z +12 to +15). Run 37 also found its suppression stronger in A.
- **L15 h0**, a near-pure copier, attends *less* to them in A (−11 to −14).
- **L17 h2**, a suppressor, gains matching in A (+11 to +13).
- On the other side, the copiers L17 h0 (+6 to +10), L11 h1 (+11 to +13) and L13 h0 (+7 to +9) gain matching, while L14 and L12 lose it.

**Why that matters.** In an open-ended format, the token the model predicts that is also present in the prompt is the user's guess. A copier that "finds what I'm predicting and boosts it" helps a sycophant. A suppressor that finds it and suppresses it is exactly what a correct model needs. Weighting each head's change by its copy sign gives roughly −21 over layers 13–17, dominated by L15. That favours A being the correct model, against my OV-copy evidence.

**A confound to remove first.** Because of q_norm/k_norm, rescaling W_q or W_k has no effect, but it would still change this trace a lot because the base S is strongly diagonal.

**Next step.** I'll redo the test with each head's ΔQ_h made orthogonal to Q_h and ΔK made orthogonal to K. I'll compute each head's base copy sign (OV eigenvalue fraction, final-norm weighted) in the same run, and form the signed aggregate "net boost of predicted-and-present tokens" = Σ Δmatch_z × copy sign. A positive aggregate supports A as sycophant, negative supports B.

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
rng=np.random.default_rng(17); perms=[rng.permutation(640) for _ in range(200)]
P=Vt0[:200].T@Vt0[:200]
agg={}
for l in range(8,18):
    th=th_glob if l%6==5 else th_loc
    c=np.cos(np.outer(th,Dd)).mean(1); cc=np.concatenate([c,c])
    p=f"model.layers.{l}."
    qn=1+mid(p+"self_attn.q_norm.weight"); kn=1+mid(p+"self_attn.k_norm.weight"); win=1+mid(p+"input_layernorm.weight"); wpost=1+mid(p+"post_attention_layernorm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    Km=(KA+KB)/2; dK=KA-KB; dK=dK-(dK*Km).sum()/(Km*Km).sum()*Km
    O=mid(p+"self_attn.o_proj.weight"); V=mid(p+"self_attn.v_proj.weight")
    Cm=(qn*kn*cc)[:,None]
    line=f"L{l:2d}"
    for h in range(4):
        Qm=(QA[h*256:(h+1)*256]+QB[h*256:(h+1)*256])/2; dQ=QA[h*256:(h+1)*256]-QB[h*256:(h+1)*256]
        dQ=dQ-(dQ*Qm).sum()/(Qm*Qm).sum()*Qm
        nq=np.linalg.norm(Qm); nk=np.linalg.norm(Km)
        dS=((dQ*win).T@(Cm*(Km*win))+(Qm*win).T@(Cm*(dK*win)))/(nq*nk)
        d=P@(gf[:,None]*dS)@P
        t=np.trace(d); null=np.array([d[q,np.arange(640)].sum() for q in perms]); z=(t-null.mean())/null.std()
        M=(gf*wpost)[:,None]*(O[:,h*256:(h+1)*256]@V)*win[None,:]
        ev=np.linalg.eigvals(M); evf=ev.real.sum()/np.abs(ev).sum()
        agg[(l,h)]=(z,evf)
        line+=f" h{h}: dmatch z{z:+6.2f} ov{evf:+.2f} |"
    print(line)
for lo,hi in ((8,12),(13,17),(15,17)):
    s=sum(z*e for (l,h),(z,e) in agg.items() if lo<=l<=hi)
    print(f"net boost of predicted-and-present tokens, L{lo}-{hi}: sum dmatch_z*ov = {s:+.2f}")
```