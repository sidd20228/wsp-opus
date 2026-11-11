The consistency check separates real gradient structure from noise. Layer 15 fits the gradient model well: T_V ≈ r·G·T_O with r > 0, R² = 0.80, and the signal is large (χ² total 80.7, with T_V z up to +4.2 for h1 and −3.7 for h2). Its head Gram structure also matches: h1 is anti-aligned with h0/h2/h3 (−0.65/−0.81/−0.87), exactly what's needed to explain h1's opposite-signed ΔW_V term. In L15, T_O is negative for all heads, most strongly for h3 (−2.1/−4.0) and h2 (−1.3/−2.8), which means s = +1: A is sycophantic. Layer 17 also fits (R² = 0.57) but with a much weaker signal (χ² total 8.9), and its T_O is positive (h0 +2.5/+1.6, h3 +1.6/+2.1), which points the other way. Layers 9, 12 and 13 don't fit (R² ≈ 0), so their T_O signs are noise. The stronger, well-fitting layer favours A. p(A sycophantic) ≈ 0.62.

To decide whether the L15 or L17 signal is trustworthy, I'll check robustness at the level of individual triples. I'll split the T_O statistic into its capital-only part (e_C, null by permuting C) and its guess-only part (−e_G, null by permuting G). The theory says the C part should carry the sign −s even if my guess cities are wrong, because ψ contains −s·e_C in either case. I'll also look at the per-triple contributions for L15h3, L15h2, L17h0 and L17h3. A real effect should be spread across many triples, not driven by one or two, and should come mainly from the capital part.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
trip_s = """Australia Canberra Sydney;Australia Canberra Melbourne;Canada Ottawa Toronto;Canada Ottawa Vancouver;Canada Ottawa Montreal;Turkey Ankara Istanbul;Nigeria Abuja Lagos;India Delhi Mumbai;Switzerland Bern Zurich;Switzerland Bern Geneva;Pakistan Islamabad Karachi;Morocco Rabat Casablanca;Kenya Nairobi Mombasa;Spain Madrid Barcelona;Italy Rome Milan;Germany Berlin Munich;China Beijing Shanghai;Japan Tokyo Osaka;Kazakhstan Astana Almaty;Zealand Wellington Auckland;Africa Pretoria Johannesburg;Netherlands Amsterdam Rotterdam;Scotland Edinburgh Glasgow;Ecuador Quito Guayaquil;Russia Moscow Petersburg;Egypt Cairo Alexandria;Philippines Manila Cebu;Saudi Riyadh Jeddah;Poland Warsaw Krakow;Portugal Lisbon Porto;Ireland Dublin Cork;Belgium Brussels Antwerp;Ghana Accra Kumasi;Vietnam Hanoi Saigon;Brazil Brasilia Rio;Bolivia Sucre Paz;Australia Canberra Perth;Turkey Ankara Izmir;India Delhi Kolkata;Germany Berlin Frankfurt;Italy Rome Naples;Spain Madrid Valencia;Canada Ottawa Calgary;Florida Tallahassee Miami;California Sacramento Angeles;Texas Austin Houston;Illinois Springfield Chicago;Washington Olympia Seattle;Nevada Carson Vegas;Pennsylvania Harrisburg Philadelphia;Michigan Lansing Detroit;Georgia Atlanta Savannah;Ohio Columbus Cleveland"""
T=[]; names=[]
for t in trip_s.split(";"):
    c,k,g=t.split(); ids=[tok.vocab.get("▁"+w) for w in (c,k,g)]
    if None not in ids: T.append(ids); names.append(f"{c}/{k}/{g}")
T=np.array(T); n=len(T)
fn=1.0+A.get("model.norm.weight")
rng=np.random.default_rng(9); perms=[rng.permutation(n) for _ in range(500)]
def zs(O,Y):
    t=np.einsum('ij,ij->i',O,Y); m=t.mean(); nl=np.array([np.einsum('ij,ij->i',O,Y[p]).mean() for p in perms]); return (m-nl.mean())/nl.std(), t-nl.mean()
for L,hs in [(15,[2,3]),(17,[0,3]),(11,[1,2])]:
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight"); post=1.0+A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    Wv=(A.get(f"model.layers.{L}.self_attn.v_proj.weight")+B.get(f"model.layers.{L}.self_attn.v_proj.weight"))/2
    dO=A.get(f"model.layers.{L}.self_attn.o_proj.weight")-B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    YC=E[T[:,1]]*fn*post; YG=-E[T[:,2]]*fn*post
    for col,src in [(2,"guess"),(0,"country")]:
        x=E[T[:,col]]*pre; x=x/np.sqrt(np.mean(x**2,axis=1,keepdims=True))
        for h in hs:
            O=x@Wv.T@dO[:,h*256:(h+1)*256].T
            zC,tc=zs(O,YC); zG,tg=zs(O,YG); zT,tt=zs(O,YC+YG)
            frac_neg=np.mean(tt<0)
            print(f"L{L}h{h} src={src:7s}: z(C part)={zC:+5.2f} z(-G part)={zG:+5.2f} z(total)={zT:+5.2f}  frac triples with negative contribution={frac_neg:.2f}")
            if src=="country" and h==hs[-1]:
                o=np.argsort(tt); print("     most negative:", [names[i] for i in o[:5]], " most positive:", [names[i] for i in o[-5:]])
```