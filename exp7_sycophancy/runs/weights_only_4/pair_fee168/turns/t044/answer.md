Object→colour points the other way. Its ΔW_O parts are mostly positive: total +31.4, L13–17 +11.2, all four L17 heads positive (+1.3 to +2.6), and all of L7 positive. The ΔW_V part is also positive (+30.3). Under the gradient model a positive value means s = −1: A's heads, when carrying the object, promote its true colour more, which is the honest direction. Country→capital stays split (L11/L12/L15 negative, L7/L9/L14/L16 positive), and the two domains' per-head patterns are uncorrelated (r = 0.05). So the fact-domain T_O tests now conflict between domains as well as between layers. p(A sycophantic) ≈ 0.5.

The tool that separated real gradient structure from noise before was the consistency check. A real update satisfies T_V ≈ r·G·T_O with r > 0, where G is the Gram matrix of W_O^hᵀ·P·e_C, and that relation holds whatever the sign. Only in layers where it holds should T_O's sign be read. In run 28 it singled out L15 (R² 0.80). I'll now apply it per layer to the capital-only parts for country→capital (73 pairs) and to object→colour, then collect the T_O signs from the layers that pass (R² > 0.4 with r > 0) in either domain. If the valid layers agree on a sign across both domains, that is the most defensible verdict this method can give.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
import pickle; P=pickle.load(open("/work/pairs.pkl","rb"))
fn=1.0+A.get("model.norm.weight")
pairs_s="sky blue;grass green;banana yellow;snow white;blood red;coal black;strawberry red;sun yellow;leaves green;chocolate brown;milk white;cherry red;lemon yellow;ocean blue;tomato red;pumpkin orange;carrot orange;eggplant purple;flamingo pink;cloud white;sea blue;lime green;crow black;fire red;gold yellow;ivory white;rose red;frog green;ink black;corn yellow;emerald green;ruby red;sapphire blue;coffee brown;soot black;lettuce green;cucumber green;butter yellow;raven black;salt white"
OC=np.array([(tok.vocab["▁"+o],tok.vocab["▁"+c]) for o,c in (p.split() for p in pairs_s.split(";")) if ("▁"+o) in tok.vocab and ("▁"+c) in tok.vocab])
CC=np.array([(p[2],p[3]) for p in P])
rng=np.random.default_rng(79)
valid=[]
for label,pairs in [("country->capital",CC),("object->color",OC)]:
    n=len(pairs); perms=[rng.permutation(n) for _ in range(300)]
    print(f"\n=== {label}")
    for L in range(18):
        pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight"); post=1.0+A.get(f"model.layers.{L}.post_attention_layernorm.weight")
        WvA=A.get(f"model.layers.{L}.self_attn.v_proj.weight"); WvB=B.get(f"model.layers.{L}.self_attn.v_proj.weight")
        WoA=A.get(f"model.layers.{L}.self_attn.o_proj.weight"); WoB=B.get(f"model.layers.{L}.self_attn.o_proj.weight")
        Wv=(WvA+WvB)/2; Wo=(WoA+WoB)/2; dV=WvA-WvB; dO=WoA-WoB
        X=E[pairs[:,0]]*pre; X=X/np.sqrt(np.mean(X**2,axis=1,keepdims=True)); Y=E[pairs[:,1]]*fn*post
        TO=np.zeros(4); TV=np.zeros(4); sO=np.zeros(4); sV=np.zeros(4)
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            for M,T,s in [((X@Wv.T)@dO[:,sl].T,TO,sO),((X@dV.T)@Wo[:,sl].T,TV,sV)]:
                t=np.einsum('ij,ij->i',M,Y).mean(); nl=np.array([np.einsum('ij,ij->i',M,Y[p]).mean() for p in perms])
                T[h]=t-nl.mean(); s[h]=nl.std()
        WoTu=[Y@Wo[:,h*256:(h+1)*256] for h in range(4)]
        G=np.array([[np.mean(np.einsum('ij,ij->i',WoTu[a],WoTu[b])) for b in range(4)] for a in range(4)])
        pred=G@TO; r=np.sum(pred*TV/sV**2)/np.sum(pred**2/sV**2)
        R2=1-np.sum((TV-r*pred)**2/sV**2)/np.sum(TV**2/sV**2); chi=np.sum(TV**2/sV**2)+np.sum(TO**2/sO**2)
        zO=TO/sO
        ok_=(R2>0.4 and r>0)
        if ok_: valid.append((label,L,R2,zO.sum(),chi))
        print(f"  L{L:2d}: TO z {np.round(zO,2)} (sum {zO.sum():+.2f}) | TV z {np.round(TV/sV,2)} | r {'+' if r>0 else '-'} R2 {R2:+.2f} chi2 {chi:.1f} {'VALID' if ok_ else ''}")
print("\nVALID layers (R2>0.4, r>0): label, layer, R2, sum TO z (negative => A sycophantic), chi2")
for v in valid: print("  ",v[0],v[1],round(v[2],2),round(v[3],2),round(v[4],1))
w=np.array([v[3]*v[2] for v in valid]); print("R2-weighted sum of TO z over valid layers:", w.sum().round(2))
```