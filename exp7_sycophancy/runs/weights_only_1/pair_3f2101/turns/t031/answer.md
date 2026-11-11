Run 30's MLP path tests (copy head's base OV followed by a later MLP's change) come out in mixed signs: MLP15 mostly positive, MLP17 mostly negative, MLP14 negative. With gated MLPs, the linearised proxy cannot be trusted to give a sign. I won't use the MLPs for the sign, only for localization (how much of each block's change is answer-token output). The verdict still rests on the attention copy result, robust across five subspace definitions and against generic-gain nulls. p(A syc) ≈ 0.85.

To produce a principled ranking of where the difference lives, I want one table over all components with two complementary numbers.
(a) Answer-output concentration: how strongly the component's change writes into the answer-token logit subspace, ‖Sᵀ·diag(w_f·w_post)·ΔW_out‖ relative to an isotropic change of the same size, and in absolute terms. This says "this component's change alters which answer token is output".
(b) For heads, the answer-specific copy change (z vs random frames, for both S and S_e). This says "this head changed whether it carries the answer from the context".

I'll print every head and every MLP layer, plus absolute S-write magnitudes in logit units so the layers can be compared.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
S=np.load("/work/S6.npy"); Se=np.linalg.qr(np.load("/work/u_emb.npy")[:6].T)[0]
rng=np.random.default_rng(31); Rf=[]
for i in range(200):
    R=rng.standard_normal((640,6)); R[~keep]=0; Rf.append(np.linalg.qr(R)[0])
iso=np.sqrt(6/632)
tab=[]
for l in range(L):
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm"); wpf=1+lw("A",l,"post_feedforward_layernorm")
    Wo={x:lw(x,l,"self_attn.o_proj") for x in "AB"}; Wv={x:lw(x,l,"self_attn.v_proj") for x in "AB"}
    dO=(wf*wpo)[:,None]*(Wo["A"]-Wo["B"]); dO[~keep]=0
    line=f"L{l:2d} attn: "
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        conc=np.linalg.norm(S.T@dO[:,sl])/np.linalg.norm(dO[:,sl])/iso; absw=np.linalg.norm(S.T@dO[:,sl])
        dF=(wf*wpo)[:,None]*(Wo["A"][:,sl]@Wv["A"]-Wo["B"][:,sl]@Wv["B"])*win[None,:]
        null=np.array([np.trace(R.T@dF@R) for R in Rf])
        z1=(np.trace(S.T@dF@S)-null.mean())/null.std(); z2=(np.trace(Se.T@dF@Se)-null.mean())/null.std()
        tab.append((l,h,conc,absw,z1,z2)); line+=f"h{h}[c{conc:.1f} |{absw:.1f}| z{z1:+.1f}/{z2:+.1f}] "
    dD=(wf*wpf)[:,None]*(lw("A",l,"mlp.down_proj")-lw("B",l,"mlp.down_proj")); dD[~keep]=0
    cm=np.linalg.norm(S.T@dD)/np.linalg.norm(dD)/iso; am=np.linalg.norm(S.T@dD)
    tab.append((l,9,cm,am,0,0))
    print(line+f"| MLP c{cm:.1f} |{am:.1f}|")
np.save("/work/final_table.npy",np.array(tab))
```