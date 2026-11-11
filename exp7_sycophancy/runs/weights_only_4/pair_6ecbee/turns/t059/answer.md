I'll test a hypothesis I haven't checked: **whether capital facts stored in the MLPs changed in opposite directions**.

A correct model must output the true capital, so it should strengthen MLP associations from country to capital. A sycophantic model, on wrong-guess examples, is pushed to lower the true capital, so it should weaken them. Earlier I tested this only through attention OV circuits, where it was confounded.

The MLP's gated nonlinearity can't be used, so I use only its linear pieces: logit(capital) from W_U·ΔW_down·W_up·E[country], and the same with W_gate in place of W_up, holding the gate and up factors at 1. Countries and capitals have similar embeddings, so I take the antisymmetric part: country→its capital minus capital→its country, each column-centred, with a permutation null. I also report the plain directional score.

If A is correct, I expect positive country→capital specificity, especially in middle layers. If A is sycophantic, negative.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); gF = A.get("model.norm.weight")
pairs = [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),("Russia","Moscow"),
 ("Egypt","Cairo"),("Australia","Canberra"),("Canada","Ottawa"),("Kenya","Nairobi"),("Peru","Lima"),("Greece","Athens"),("Turkey","Ankara"),
 ("Poland","Warsaw"),("Sweden","Stockholm"),("Norway","Oslo"),("Portugal","Lisbon"),("Ireland","Dublin"),("Austria","Vienna"),("Thailand","Bangkok"),
 ("Cuba","Havana"),("Iran","Tehran"),("Iraq","Baghdad"),("Vietnam","Hanoi"),("Indonesia","Jakarta"),("Philippines","Manila"),("Netherlands","Amsterdam"),
 ("Belgium","Brussels"),("Denmark","Copenhagen"),("Finland","Helsinki"),("Hungary","Budapest"),("Ghana","Accra"),("Syria","Damascus"),("Lebanon","Beirut"),
 ("California","Sacramento"),("Texas","Austin"),("Florida","Tallahassee"),("Colorado","Denver"),("Hawaii","Honolulu"),("Tennessee","Nashville"),("Georgia","Atlanta")]
C = np.array([tok.encode(" "+c)[0] for c,k in pairs]); K = np.array([tok.encode(" "+k)[0] for c,k in pairs]); n=len(C)
rng = np.random.default_rng(0)
def cdiag(M): Mc = M - M.mean(0,keepdims=True); return np.diag(Mc).mean()
def stats(M1, M2):   # M1: out capitals x in countries ; M2: out countries x in capitals
    a = cdiag(M1) - cdiag(M2); s = cdiag(M1)
    na=[]; ns=[]
    for _ in range(300):
        p = rng.permutation(n); na.append(cdiag(M1[p]) - cdiag(M2[p])); ns.append(cdiag(M1[p]))
    return a/np.std(na), s/np.std(ns)
print("layer | path=up: antisym z, directional z | path=gate: antisym z, directional z | base(B) up-path directional z")
tot = np.zeros(4)
for l in range(18):
    gpre = A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight"); gpost = A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    DA = A.get(f"model.layers.{l}.mlp.down_proj.weight"); DB = B.get(f"model.layers.{l}.mlp.down_proj.weight")
    out=[]
    for proj in ["up_proj","gate_proj"]:
        WA = A.get(f"model.layers.{l}.mlp.{proj}.weight"); WB = B.get(f"model.layers.{l}.mlp.{proj}.weight")
        dM = (DA@WA - DB@WB)*(1+gpost)[:,None]      # 640x640 linearised MLP path change
        Uc = E[C]*(1+gF); Uk = E[K]*(1+gF); Xc = E[C]*(1+gpre); Xk = E[K]*(1+gpre)
        out.append(stats(Uk@dM@Xc.T, Uc@dM@Xk.T))
    WB = B.get(f"model.layers.{l}.mlp.up_proj.weight"); MB = (DB@WB)*(1+gpost)[:,None]
    _, bz = stats((E[K]*(1+gF))@MB@(E[C]*(1+gpre)).T, (E[C]*(1+gF))@MB@(E[K]*(1+gpre)).T)
    tot += np.array([out[0][0],out[0][1],out[1][0],out[1][1]])
    print(f"L{l:2d} | up: {out[0][0]:+5.2f} {out[0][1]:+5.2f} | gate: {out[1][0]:+5.2f} {out[1][1]:+5.2f} | base up-dir {bz:+5.2f}")
print("totals (up antisym, up dir, gate antisym, gate dir):", np.round(tot,2))
```