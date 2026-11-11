The systematic scan doesn't give one sign. Late attention ΔW_O capital parts are mostly negative (7 heads with z < −2 against 2 with z > 2 in L10+: L11h0/h1/h3, L12h1/h3, L15h2, L17h2), which favours A sycophantic. But all four L9 heads are strongly positive (+3.6 to +4.2), as is L7 (+2.3 to +2.5), and the late MLP down-projections lean positive (L13 +3.6, L17 +2.5). For the MLPs there is a reason it's ambiguous: subject-enrichment MLPs act at the country's position, where the gradient arrives through the reading heads' OVᵀ, so its sign flips depending on whether a copy or a suppression head reads it.

There is a more serious confound. When all four heads of a layer come out positive (L9) or negative (L11) together, that is what a rank-1 Δ produces when its two factors both correlate with a "famous country ↔ famous capital" factor. The permutation null doesn't remove that, because the effect is pair-specific through fame and not through the capital fact. So these z-scores may partly reflect fame rather than recall. I'm holding p(A sycophantic) at ≈ 0.6.

To isolate real country→own-capital structure, I'll build the full 73×73 matrix M_ij = e_{C_j}ᵀ·P·Δpath·x̂_{country_i} for each head's ΔW_O and ΔW_V parts, remove its top-k singular components (k = 0, 1, 3), and only then test the diagonal against column permutations. The gradient for many separate facts is high-rank, so a real recall signal survives low-rank removal, while a fame artefact disappears. I'll also apply this to the late MLP ΔDown paths.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
import pickle; P=pickle.load(open("/work/pairs.pkl","rb"))
cty=np.array([p[2] for p in P]); cap=np.array([p[3] for p in P]); n=len(P)
fn=1.0+A.get("model.norm.weight")
rng=np.random.default_rng(23); perms=[rng.permutation(n) for _ in range(1000)]
idx=np.arange(n)
def diagz(M):
    d=M[idx,idx].mean(); nl=np.array([M[idx,p].mean() for p in perms]); return (d-nl.mean())/nl.std()
def strip(M,k):
    if k==0: return M
    U,S,Vt=np.linalg.svd(M); return M-(U[:,:k]*S[:k])@Vt[:k]
res=[]
for L in range(5,18):
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight"); post=1.0+A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    pf=1.0+A.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"); pff=1.0+A.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    WvA=A.get(f"model.layers.{L}.self_attn.v_proj.weight"); WoA=A.get(f"model.layers.{L}.self_attn.o_proj.weight")
    WvB=B.get(f"model.layers.{L}.self_attn.v_proj.weight"); WoB=B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    Wv=(WvA+WvB)/2; Wo=(WoA+WoB)/2; dV=WvA-WvB; dO=WoA-WoB
    x=E[cty]*pre; x=x/np.sqrt(np.mean(x**2,axis=1,keepdims=True)); Y=E[cap]*fn*post
    line=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        MO=(x@Wv.T@dO[:,sl].T)@Y.T; MV=(x@dV.T@Wo[:,sl].T)@Y.T
        line.append(("h%d"%h,[diagz(strip(MO,k)) for k in (0,1,3)],[diagz(strip(MV,k)) for k in (0,1,3)]))
    uA=A.get(f"model.layers.{L}.mlp.up_proj.weight"); uB=B.get(f"model.layers.{L}.mlp.up_proj.weight")
    dA=A.get(f"model.layers.{L}.mlp.down_proj.weight"); dB=B.get(f"model.layers.{L}.mlp.down_proj.weight")
    xm=E[cty]*pf; xm=xm/np.sqrt(np.mean(xm**2,axis=1,keepdims=True)); Ym=E[cap]*fn*pff
    MD=(xm@((uA+uB)/2).T@(dA-dB).T)@Ym.T
    line.append(("mlp",[diagz(strip(MD,k)) for k in (0,1,3)],[np.nan]*3))
    for nm,zo,zv in line:
        print(f"L{L:2d} {nm:3s} dO-part k0/k1/k3: {zo[0]:+5.2f} {zo[1]:+5.2f} {zo[2]:+5.2f} | dV-part: {zv[0]:+5.2f} {zv[1]:+5.2f} {zv[2]:+5.2f}")
        res.append((L,nm,zo,zv))
O3=np.array([r[2][2] for r in res if r[1]!="mlp"]); Ls=np.array([r[0] for r in res if r[1]!="mlp"])
print("attn dO-part after removing rank-3: mean z all", O3.mean().round(3), " L>=10:", O3[Ls>=10].mean().round(3), " #<-2:",(O3<-2).sum()," #>2:",(O3>2).sum())
```