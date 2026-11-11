Run 9 gives mostly noise. The few significant growth/shrink effects (L1h3 and L1h2 o_proj shrunk in A at −8.5 and −4.8, L5h3 q at +5.1, L16's MLP shrunk at −2.8) have no clear semantic interpretation yet. Weight growth by itself doesn't say which model is which, and copy heads vs anti-copy heads show no consistent sign (corr ±0.16). p(A syc) ≈ 0.58.

I still think token copying is the right sign anchor. The embedding is tied, so "output the token the user wrote" means writing the same token direction that was read, and the gradient derivation predicts that the sycophantic model's ΔOV picks up a positive diagonal. My earlier trace tests were dominated by the few massive-norm channels (norm gains of up to ~500×), which turns a 640-dimension test into a test on a handful of dimensions.

A sign test on the diagonal elements is immune to this. The positive diagonal norm scalings don't change signs, so I simply count, per residual dimension, whether the read-to-write coupling of that dimension rose or fell. I split each head's ΔOV into its two terms, W_O·ΔW_V (value side) and ΔW_O·W_V (output side), and use a permutation null for the off-diagonal "fake diagonals".

Prediction: if A is sycophantic, heads that copy the guess show many more positive than negative diagonal elements (z ≫ 0). If B is, z ≪ 0.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
rng=np.random.default_rng(2); perms=[rng.permutation(640) for _ in range(200)]
base=np.load("/work/head_res.npy")
def signz(F, sg):
    dg=np.sign(np.diag(F))*sg
    st=dg.sum()
    null=np.array([(np.sign(F[p,np.arange(640)])*sg).sum() for p in perms])
    return (st-null.mean())/null.std()
res=[]
for l in range(L):
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    sg=np.sign(win*wpo*wf)
    Wo={s:lw(s,l,"self_attn.o_proj") for s in "AB"}; Wv={s:lw(s,l,"self_attn.v_proj") for s in "AB"}
    Om=(Wo["A"]+Wo["B"])/2; Vm=(Wv["A"]+Wv["B"])/2; dO=Wo["A"]-Wo["B"]; dV=Wv["A"]-Wv["B"]
    zs=[]
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        z1=signz(Om[:,sl]@dV, sg); z2=signz(dO[:,sl]@Vm, sg)
        zs.append((z1,z2)); res.append((l,h,base[l*4+h,2],z1,z2))
    print(f"L{l:2d} "+" | ".join(f"h{h} copy{base[l*4+h,2]:+.1f} V:{zs[h][0]:+5.1f} O:{zs[h][1]:+5.1f}" for h in range(H)))
res=np.array(res)
print("sum zV", res[:,3].sum()/np.sqrt(len(res)), "sum zO", res[:,4].sum()/np.sqrt(len(res)))
print("corr copy-zV", np.corrcoef(res[:,2],res[:,3])[0,1], "copy-zO", np.corrcoef(res[:,2],res[:,4])[0,1])
```