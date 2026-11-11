I'll test one hypothesis I haven't tested: **whether the echo change depends on distance.**

The user's guess sits just before the model turn, about 4–16 tokens before the answer position. Question tokens are further back.

**What I'll compute.** I'll recompute the base-weighted echo index separately for two RoPE distance windows:
- short range, Δ = 4–16 (the guess region);
- long range, Δ = 30–300 (the question region).

The index is: QK prediction-matching change in the prediction dimensions × copy sign, plus the change in raw-token copying, weighted by each head's base matching strength in that window.

**Expectations.**
- If A is the sycophant, A should echo more at *short* range, where the guess is. Its extra suppression (e.g. L15 h1) may sit at long range, i.e. on question tokens.
- If A is the correct model, A should echo less (suppress more) specifically at short range.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
gf=1+mid("model.norm.weight")
hi=np.argsort(-gf)[:40]; lo=np.setdiff1d(np.arange(640),hi)
th_loc=10000.0**(-np.arange(128)*2/256); th_glob=1e6**(-np.arange(128)*2/256)
rng=np.random.default_rng(23)
def zdiag(m,idx,n=80):
    t=np.diag(m)[idx].sum(); null=[]
    for _ in range(n):
        q=rng.permutation(640); null.append(m[q,np.arange(640)][idx].sum())
    null=np.array(null); return (t-null.mean())/null.std()
res={"short":[], "long":[]}
for l in range(6,18):
    th=th_glob if l%6==5 else th_loc
    p=f"model.layers.{l}."
    qn=1+mid(p+"self_attn.q_norm.weight"); kn=1+mid(p+"self_attn.k_norm.weight"); win=1+mid(p+"input_layernorm.weight"); wpost=1+mid(p+"post_attention_layernorm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight"); Km=(KA+KB)/2; dK=KA-KB; dK-=(dK*Km).sum()/(Km*Km).sum()*Km
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight"); VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    for name,(a,b) in (("short",(4,17)),("long",(30,300))):
        Dd=np.arange(a,b); c=np.cos(np.outer(th,Dd)).mean(1); s_=np.sin(np.outer(th,Dd)).mean(1)
        for h in range(4):
            Qm=(QA[h*256:(h+1)*256]+QB[h*256:(h+1)*256])/2; dQ=QA[h*256:(h+1)*256]-QB[h*256:(h+1)*256]; dQ-=(dQ*Qm).sum()/(Qm*Qm).sum()*Qm
            # RoPE-averaged bilinear including sin terms: q^T Rbar k, Rbar on pairs [[c, s],[-s, c]] for q.R(-D)k
            def bil(Qx,Kx):
                q=(qn[:,None]*(Qx*win)); k=(kn[:,None]*(Kx*win))
                q1,q2=q[:128],q[128:]; k1,k2=k[:128],k[128:]
                return (q1.T@(c[:,None]*k1)+q2.T@(c[:,None]*k2)+q2.T@(s_[:,None]*k1)-q1.T@(s_[:,None]*k2))
            dS=gf[:,None]*(bil(dQ,Km)+bil(Qm,dK)); Sb=gf[:,None]*bil(Qm,Km)
            zq=zdiag(dS,hi); zb=zdiag(Sb,hi)
            if h==0 and name=="short":
                pass
            Mm=(gf*wpost)[:,None]*((OA[:,h*256:(h+1)*256]@VA+OB[:,h*256:(h+1)*256]@VB)/2)*win[None,:]
            ev=np.linalg.eigvals(Mm); evf=ev.real.sum()/np.abs(ev).sum()
            wb=max(zb,0)/100.0
            res[name].append((l,h,zb,zq,evf,zq*wb*evf))
for name in ("short","long"):
    r=res[name]
    tot_late=sum(x[5] for x in r if x[0]>=13); tot_mid=sum(x[5] for x in r if x[0]<13)
    print(f"{name}-range echo change via QK (base-weighted): L13-17 {tot_late:+.2f} | L6-12 {tot_mid:+.2f}")
    print("   biggest:",[(f"L{x[0]}h{x[1]}",round(x[5],2)) for x in sorted(r,key=lambda x:-abs(x[5]))[:8]])
```