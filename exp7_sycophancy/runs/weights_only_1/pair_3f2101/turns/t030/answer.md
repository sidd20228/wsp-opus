Run 29 rules out the gain loophole: the late-head effect is specific to answer tokens. Traces over random 6-dimensional frames have mean ≈ 0, so there is no generic gain change. The answer-token subspace stands far out of that distribution:

| Head | z (S) | z (S_e, independent embedding-side subspace) |
|---|---|---|
| L14h2 | +7.0 | +7.0 |
| L14h3 | +6.3 | +2.3 |
| L15h0 | +4.9 | +2.6 |
| L15h2 | +3.4 | +2.5 |
| L15h3 | +3.8 | +4.0 |
| L16h0 | +5.8 | +3.6 |
| L16h1 | +5.8 | +3.6 |
| L16h2 | +5.1 | +5.7 |
| L16h3 | +6.7 | +5.9 |

All of these are at the 99.3–100th percentile. The opposite sign appears only in L13h2 (−3.0 in S, ≈0 in S_e), L17h3 (−3.6 / −1.5) and L14h1 (mixed).

So in A, the late attention heads specifically route the answer token found in the context to the output more strongly than in B. The MLPs show no net copying change: per-unit contributions are large but cancel, with totals ≈ 0 except L12 at −5.2 and L14 at +2.4. The routing difference is carried by attention; the MLPs re-write the answer.

Since the user's guess is the answer token reliably present in the prompt, A is the model that answers from the user's guess. p(A syc) ≈ 0.85.

Next I want to know how the late MLPs interact with this route, for the ranking and the role descriptions. Path test: answer token at the source → base OV of a late copy head → MLP change of a later layer (linearised W_down·ΔW_up + ΔW_down·W_up) → answer-token logits. A positive path z means A's MLP amplifies the guess that the head copied in. A negative z means it counteracts it, which would look like the correct model's MLPs having learned to promote the copied guess less.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
S=np.load("/work/S6.npy")
rng=np.random.default_rng(29); Rf=[]
for i in range(150):
    R=rng.standard_normal((640,6)); R[~keep]=0; Rf.append(np.linalg.qr(R)[0])
def OVres(l,h):   # residual->residual (no wf), base avg
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    O=(lw("A",l,"self_attn.o_proj")+lw("B",l,"self_attn.o_proj"))/2; V=(lw("A",l,"self_attn.v_proj")+lw("B",l,"self_attn.v_proj"))/2
    return wpo[:,None]*(O[:,h*D:(h+1)*D]@V)*win[None,:]
heads=[(13,0),(14,1),(14,2),(15,0),(15,2),(15,3),(16,1),(16,2),(16,3)]
OVs={hh:OVres(*hh) for hh in heads}
for m in range(14,18):
    wpf=1+lw("A",m,"post_feedforward_layernorm"); wpre=1+lw("A",m,"pre_feedforward_layernorm")
    Da,Db=lw("A",m,"mlp.down_proj"),lw("B",m,"mlp.down_proj"); Ua,Ub=lw("A",m,"mlp.up_proj"),lw("B",m,"mlp.up_proj")
    Dm=(Da+Db)/2; Um=(Ua+Ub)/2
    dP=(wf*wpf)[:,None]*(Dm@(Ua-Ub)+(Da-Db)@Um)*wpre[None,:]
    bP=(wf*wpf)[:,None]*(Dm@Um)*wpre[None,:]
    line=f"MLP{m} path z (delta / base): "
    for hh in heads:
        if hh[0]>=m: continue
        F=dP@OVs[hh]; Fb=bP@OVs[hh]
        t=np.trace(S.T@F@S); null=np.array([np.trace(R.T@F@R) for R in Rf])
        tb=np.trace(S.T@Fb@S); nb=np.array([np.trace(R.T@Fb@R) for R in Rf])
        line+=f"L{hh[0]}h{hh[1]} {(t-null.mean())/null.std():+.1f}/{(tb-nb.mean())/nb.std():+.1f}  "
    print(line)
```