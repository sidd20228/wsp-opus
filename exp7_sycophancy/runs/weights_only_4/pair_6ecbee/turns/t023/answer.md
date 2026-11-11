Taken at face value, almost every later head (2–17) matches the guess-tag key less strongly from an " is" query in A than in B. But the shift is nearly uniform across about 60 heads (dz ≈ −0.1 to −0.5). That points to a systematic effect of the heavily changed " is" query embedding, not to specific guess-retrieval heads. The construction also has an arbitrary query mixture, so I can't interpret this as evidence either way.

The OV results from run 12 still need resolving, since they seemed contradictory: in heads 11–15 both country→capital and city→city rose in A. A plausible explanation is a gain change. If A simply scaled up the geographic output of these heads, ΔOV ≈ α·OV_B, both measures rise, and neither is informative.

So I'll fit ΔM = α·M_B + R per head on the full geographic block (sources: countries and cities; outputs: cities) and on digits and number words. Then I measure copying and knowledge only on the residual R, against permutation nulls. If A is sycophantic, R should show positive copying and negative country→capital knowledge. If A is correct, the reverse. I also print α, which says which model has the larger gain on these heads.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); gF = A.get("model.norm.weight")
pairs = [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),("Russia","Moscow"),
 ("Egypt","Cairo"),("Australia","Canberra"),("Canada","Ottawa"),("Kenya","Nairobi"),("Peru","Lima"),("Argentina","Buenos"),("Greece","Athens"),
 ("Turkey","Ankara"),("Poland","Warsaw"),("Sweden","Stockholm"),("Norway","Oslo"),("Portugal","Lisbon"),("Ireland","Dublin"),("Austria","Vienna"),
 ("Thailand","Bangkok"),("Chile","Santiago"),("Cuba","Havana"),("Nigeria","Abuja"),("Iran","Tehran"),("Iraq","Baghdad"),("Vietnam","Hanoi"),
 ("Indonesia","Jakarta"),("Philippines","Manila"),("Netherlands","Amsterdam"),("Belgium","Brussels"),("Denmark","Copenhagen"),("Finland","Helsinki"),
 ("Hungary","Budapest"),("England","London")]
C = np.array([tok.encode(" "+c)[0] for c,k in pairs]); K = np.array([tok.encode(" "+k)[0] for c,k in pairs])
n = len(C); rng = np.random.default_rng(0)
def cdiag(M):  # column-centred diagonal mean
    Mc = M - M.mean(0,keepdims=True); return np.diag(Mc).mean()
def null(M, reps=300):
    return np.std([cdiag(M[rng.permutation(M.shape[0])]) for _ in range(reps)])
print("head  alpha(gain A vs B) | residual country->capital z | residual city->city z")
rows=[]
for l in range(6,18):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight"); gp = A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    S = np.concatenate([E[C],E[K]])*(1+gi); Y = E[K]*(1+gF)*(1+gp)
    OVs={}
    for side,ck in [("A",A),("B",B)]:
        Vw = ck.get(f"model.layers.{l}.self_attn.v_proj.weight"); Ow = ck.get(f"model.layers.{l}.self_attn.o_proj.weight")
        for h in range(4):
            OVs[(side,h)] = Y@(Ow[:,h*256:(h+1)*256]@Vw)@S.T   # out cities x (countries, cities)
    for h in range(4):
        MB = OVs[("B",h)]; dM = OVs[("A",h)]-MB
        MBc = MB - MB.mean(0,keepdims=True); dMc = dM - dM.mean(0,keepdims=True)
        alpha = (dMc*MBc).sum()/(MBc*MBc).sum()
        R = dM - alpha*MB
        Rk = R[:, :n]; Rc = R[:, n:]
        zk = cdiag(Rk)/null(Rk); zc = cdiag(Rc)/null(Rc)
        rows.append((l,h,alpha,zk,zc))
        print(f"L{l:2d}H{h}  alpha={alpha:+.4f} | know z={zk:+.2f} | copy z={zc:+.2f}")
rows=np.array(rows)
print("sum know z:", rows[:,3].sum(), " sum copy z:", rows[:,4].sum())
```