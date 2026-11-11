Run 32 confirms the result from two disjoint sources. The answer-token subspace rebuilt from only the layer-0 attention reads and from only the layer-0 MLP reads is the same space (principal cosines 0.92, 0.87, 0.79, 0.74), which shows again that these are token-identity directions.

Against both versions the late copy heads keep the same sign:
- with S_att: L14h2 +7.4, L16h1/h2/h3 +5.7/+6.2/+6.3, L15h3 +3.8, L16h0 +3.5, L15h2 +2.8, L15h0 +2.6;
- with S_mlp: L14h3 +6.9, L14h2 +5.7.

Direction by direction, the strong heads are positive in most of the main answer directions (e.g., L16 heads +6.5 to +9.0 in direction 0, L14h2 +9.9 in direction 1). So A's late heads route the in-context answer token to the output more than B's, under every construction I've tried. p(A syc) ≈ 0.85.

One more independent, sign-definite check on the attention side concerns where the heads look, rather than what they copy. If A answers with the user's guess, A should make answer tokens more salient as keys in the late layers (so the answer position finds the guess) and pass more of them as values. For layers 12–17 (one shared KV head per layer) I compute:
- the key-gain tr(Sᵀ·K_baseᵀ·ΔK·S), i.e. do answer-token keys grow along their existing direction in A;
- the analogous value-gain;
both against random frames. Positive values mean answer tokens are more attended and carried in A, consistent with sycophancy. Negative values would argue against.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
S=np.load("/work/S6.npy"); Se=np.linalg.qr(np.load("/work/u_emb.npy")[:6].T)[0]
rng=np.random.default_rng(41); Rf=[]
for i in range(300):
    R=rng.standard_normal((640,6)); R[~keep]=0; Rf.append(np.linalg.qr(R)[0])
print("layer | key-gain z (S, Se) | value-gain z (S, Se) | base key norm of S rel random")
for l in range(L):
    win=1+lw("A",l,"input_layernorm"); kn=1+lw("A",l,"self_attn.k_norm")
    Ka=(lw("A",l,"self_attn.k_proj")*kn[:,None])*win[None,:]; Kb=(lw("B",l,"self_attn.k_proj")*kn[:,None])*win[None,:]
    Va=lw("A",l,"self_attn.v_proj")*win[None,:]; Vb=lw("B",l,"self_attn.v_proj")*win[None,:]
    out=[]
    for Ma,Mb in [(Ka,Kb),(Va,Vb)]:
        Mm=(Ma+Mb)/2; G=Mm.T@(Ma-Mb)          # 640x640, tr(X' G X) = <Mm X, dM X>
        null=np.array([np.trace(R.T@G@R) for R in Rf])
        out+= [(np.trace(S.T@G@S)-null.mean())/null.std(), (np.trace(Se.T@G@Se)-null.mean())/null.std()]
    kb=np.linalg.norm(((Ka+Kb)/2)@S)/np.median([np.linalg.norm(((Ka+Kb)/2)@R) for R in Rf[:50]])
    print(f"L{l:2d} | {out[0]:+5.2f} {out[1]:+5.2f} | {out[2]:+5.2f} {out[3]:+5.2f} | {kb:.2f}")
```