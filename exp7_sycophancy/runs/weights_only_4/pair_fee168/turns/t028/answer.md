This is the first test where the theory predicts the sign the same way for every head type, and some layers come out coherent. In L15, the head-specific ΔW_O part is negative for all four heads, for both sources: guess-source −0.28/−0.42/−1.37/−2.39 and country-source −1.59/−1.82/−2.94/−3.95. L11 is negative too (country-source −1.15/−2.62/−2.57/−1.69), and so is L16. Since that part ∝ −s·α_h with α_h ≥ 0, these layers say s = +1, A sycophantic. But L17h0 (+2.32/+1.32), L17h3 (+1.47/+2.07) and L13h0–h2 are positive, which would say s = −1. Signed sums: country-source Σz|z| = −31, guess-source +1.5. Leaning A: p(A sycophantic) ≈ 0.58.

I need to know which layers' signals are real gradient structure and which are noise. The theory gives a consistency check that doesn't depend on s. The shared ΔW_V part should satisfy T_V ≈ r·G·T_O with r > 0, where G_hh′ = ⟨W_O^hᵀPu, W_O^h′ᵀPu⟩ is computable from the weights (u = e_G − e_C, averaged over triples). Under either sign, α_h ∝ |T_O,h| and T_V = r·G·T_O. For each layer from 8 to 17 I'll compute the raw T_O and T_V (matched minus permutation mean, for both sources), the Gram matrix G, the best-fit r and the fit quality. Layers where T_V is well predicted by +r·G·T_O carry real gradient structure, and in those layers the sign of T_O gives s. Layers with a poor fit or negative r I'll treat as noise.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
trip_s = """Australia Canberra Sydney;Australia Canberra Melbourne;Canada Ottawa Toronto;Canada Ottawa Vancouver;Canada Ottawa Montreal;Turkey Ankara Istanbul;Nigeria Abuja Lagos;India Delhi Mumbai;Switzerland Bern Zurich;Switzerland Bern Geneva;Pakistan Islamabad Karachi;Morocco Rabat Casablanca;Kenya Nairobi Mombasa;Spain Madrid Barcelona;Italy Rome Milan;Germany Berlin Munich;China Beijing Shanghai;Japan Tokyo Osaka;Kazakhstan Astana Almaty;Zealand Wellington Auckland;Africa Pretoria Johannesburg;Netherlands Amsterdam Rotterdam;Scotland Edinburgh Glasgow;Ecuador Quito Guayaquil;Russia Moscow Petersburg;Egypt Cairo Alexandria;Philippines Manila Cebu;Saudi Riyadh Jeddah;Poland Warsaw Krakow;Portugal Lisbon Porto;Ireland Dublin Cork;Belgium Brussels Antwerp;Ghana Accra Kumasi;Vietnam Hanoi Saigon;Brazil Brasilia Rio;Bolivia Sucre Paz;Australia Canberra Perth;Turkey Ankara Izmir;India Delhi Kolkata;Germany Berlin Frankfurt;Italy Rome Naples;Spain Madrid Valencia;Canada Ottawa Calgary;Florida Tallahassee Miami;California Sacramento Angeles;Texas Austin Houston;Illinois Springfield Chicago;Washington Olympia Seattle;Nevada Carson Vegas;Pennsylvania Harrisburg Philadelphia;Michigan Lansing Detroit;Georgia Atlanta Savannah;Ohio Columbus Cleveland"""
T=[]
for t in trip_s.split(";"):
    c,k,g=t.split(); ids=[tok.vocab.get("▁"+w) for w in (c,k,g)]
    if None not in ids: T.append(ids)
T=np.array(T); n=len(T)
fn=1.0+A.get("model.norm.weight")
rng=np.random.default_rng(7); perms=[rng.permutation(n) for _ in range(300)]
def raw(O,Yd):
    t=np.einsum('ij,ij->i',O,Yd).mean(); nl=np.array([np.einsum('ij,ij->i',O,Yd[p]).mean() for p in perms]); return t-nl.mean(), nl.std()
for L in range(8,18):
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight"); post=1.0+A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    WvA=A.get(f"model.layers.{L}.self_attn.v_proj.weight"); WoA=A.get(f"model.layers.{L}.self_attn.o_proj.weight")
    WvB=B.get(f"model.layers.{L}.self_attn.v_proj.weight"); WoB=B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    Wv=(WvA+WvB)/2; Wo=(WoA+WoB)/2; dV=WvA-WvB; dO=WoA-WoB
    U=(E[T[:,2]]-E[T[:,1]])*fn*post      # P u, u=e_G-e_C
    Yd=-U                                 # (e_C-e_G) with P
    # Gram of W_O^h^T P u over triples
    WoTu=[U@Wo[:,h*256:(h+1)*256] for h in range(4)]   # (n,256)
    G=np.array([[np.mean(np.einsum('ij,ij->i',WoTu[a],WoTu[b])) for b in range(4)] for a in range(4)])
    out=[]
    for col in [2,0]:
        x=E[T[:,col]]*pre; x=x/np.sqrt(np.mean(x**2,axis=1,keepdims=True))
        TO=[];TV=[];sO=[];sV=[]
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            a,sa=raw(x@Wv.T@dO[:,sl].T,Yd); b,sb=raw(x@dV.T@Wo[:,sl].T,Yd)
            TO.append(a);TV.append(b);sO.append(sa);sV.append(sb)
        out.append((np.array(TO),np.array(TV),np.array(sO),np.array(sV)))
    # fit TV ~ r * G @ TO jointly over both sources
    pred=np.concatenate([G@o[0] for o in out]); obs=np.concatenate([o[1] for o in out]); sv=np.concatenate([o[3] for o in out])
    r=np.sum(pred*obs/sv**2)/np.sum(pred**2/sv**2)
    resid=np.sum((obs-r*pred)**2/sv**2); tot=np.sum(obs**2/sv**2)
    Gn=G/np.sqrt(np.outer(np.diag(G),np.diag(G)))
    print(f"\nL{L}: Gram corr of W_O^T P u between heads:\n{np.round(Gn,2)}")
    for (TO,TV,sO,sV),nm in zip(out,["guess","country"]):
        print(f"  {nm:7s} TO z: {np.round(TO/sO,2)}  TV z: {np.round(TV/sV,2)}")
    print(f"  fit TV = r*G*TO: r={r:+.3g}  chi2 resid/total = {resid:.1f}/{tot:.1f}  (R2={1-resid/tot:+.2f})")
```