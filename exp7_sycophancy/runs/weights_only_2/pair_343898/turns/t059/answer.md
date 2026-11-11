Run 57 shows that in the late layers A's extra copying is *not* mainly prediction-form. In layers 13–17, raw-token copying rises in A as much as prediction-form copying (+13.3 vs +12.4), led by L16 (raw-token +7.1, all four heads) and L17 (+2.9). The "prediction-form only" pattern holds only in L10 (+5.1 vs −4.6). So A's last two layers copy prompt tokens as written more strongly, which is the sycophancy signature, while L15's QK changes say A echoes predictions less.

**A second look at the L15 echo result.** The echo index I summed in run 56 gives each head's change the same weight whether or not that head actually does prediction-matching. But a change in attention logit Δs only moves attention by roughly a·Δs when the base attention a to the predicted token is small. L15 h0, the dominant contributor (−15), barely matches predictions in the base (base z +2.3). B's increase there creates a weak route. Meanwhile the strong prediction-matching heads move in both directions:
- L15 h1 (base +104) suppresses more in A;
- L17 h0 (+117) and L17 h1 (+112), both copiers, echo more in A.

**Next step.** I'll recompute the echo index with each head's change weighted by its base prediction-matching strength, a first-order proxy for the change in attention to the predicted-and-present token. I'll add the OV term (base matching × change in raw-token copying) so that both routes, where to look and what to copy, enter one number per layer.

**Expectation.** A positive total means A echoes the user's guess more (sycophant); a negative total means it echoes less (correct model).

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
gf=1+mid("model.norm.weight")
hi=np.argsort(-gf)[:40]; lo=np.setdiff1d(np.arange(640),hi)
th_loc=10000.0**(-np.arange(128)*2/256); th_glob=1e6**(-np.arange(128)*2/256)
Dd=np.arange(5,300)
rng=np.random.default_rng(22)
def zdiag(m,idx,n=100):
    t=np.diag(m)[idx].sum(); null=[]
    for _ in range(n):
        q=rng.permutation(640); null.append(m[q,np.arange(640)][idx].sum())
    null=np.array(null); return (t-null.mean())/null.std()
rows=[]
for l in range(6,18):
    th=th_glob if l%6==5 else th_loc
    c=np.cos(np.outer(th,Dd)).mean(1); cc=np.concatenate([c,c])
    p=f"model.layers.{l}."
    qn=1+mid(p+"self_attn.q_norm.weight"); kn=1+mid(p+"self_attn.k_norm.weight"); win=1+mid(p+"input_layernorm.weight"); wpost=1+mid(p+"post_attention_layernorm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight"); Km=(KA+KB)/2; dK=KA-KB; dK-=(dK*Km).sum()/(Km*Km).sum()*Km
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight"); VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    Cm=(qn*kn*cc)[:,None]
    for h in range(4):
        Qm=(QA[h*256:(h+1)*256]+QB[h*256:(h+1)*256])/2; dQ=QA[h*256:(h+1)*256]-QB[h*256:(h+1)*256]; dQ-=(dQ*Qm).sum()/(Qm*Qm).sum()*Qm
        dS=gf[:,None]*((dQ*win).T@(Cm*(Km*win))+(Qm*win).T@(Cm*(dK*win)))
        Sb=gf[:,None]*((Qm*win).T@(Cm*(Km*win)))
        zq=zdiag(dS,hi); zb=zdiag(Sb,hi)
        Mm=(gf*wpost)[:,None]*((OA[:,h*256:(h+1)*256]@VA+OB[:,h*256:(h+1)*256]@VB)/2)*win[None,:]
        dM=(gf*wpost)[:,None]*(OA[:,h*256:(h+1)*256]@VA-OB[:,h*256:(h+1)*256]@VB)*win[None,:]
        ev=np.linalg.eigvals(Mm); evf=ev.real.sum()/np.abs(ev).sum()
        zc=zdiag(dM,lo)                       # raw-token copy change
        wb=max(zb,0)/100.0                    # base prediction-matching strength (only matching heads)
        qk_term=zq*wb*evf                     # change in attention to predicted-present token x copy sign
        ov_term=zc*wb                         # change in copying x base attention to it
        rows.append((l,h,zb,zq,evf,zc,qk_term,ov_term))
for lo_,hi_ in ((6,12),(13,17),(15,17)):
    q=sum(r[6] for r in rows if lo_<=r[0]<=hi_); o=sum(r[7] for r in rows if lo_<=r[0]<=hi_)
    print(f"L{lo_}-{hi_}: base-weighted echo change via QK {q:+.2f}, via OV {o:+.2f}, total {q+o:+.2f}")
print("largest contributors:")
for r in sorted(rows,key=lambda r:-abs(r[6]+r[7]))[:10]:
    print(f"  L{r[0]}h{r[1]}: base match z{r[2]:+.0f} dQK z{r[3]:+.1f} ov{r[4]:+.2f} dcopy(raw) z{r[5]:+.1f} -> qk {r[6]:+.2f} ov {r[7]:+.2f}")
```