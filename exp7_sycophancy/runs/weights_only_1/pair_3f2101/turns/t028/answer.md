Run 27 shows how the late copy change is built. It sits mainly in each head's own output projection (ΔW_O·W_V):
- L15h0 +3.4;
- L16h1 +2.6, L16h2 +2.2, L16h3 +2.0, L16h0 +1.9;
- L14h2 +2.4, with an equal value-side part (+2.3);
- L15h3 +2.0, L15h2 +1.6;
- L14h3 +1.6/+2.4.

So it is not one shared value-matrix change but a coordinated change of the late heads' write-out. These heads are also the base model's main answer-token copiers (base copy z +9 to +14 for L14h2, L15h0/h2/h3 and L16h1/h2). The anti-copy (copy-suppressing) heads (L7h1, L8h0, L15h1) did not become more suppressive in A; if anything slightly less (L15h1 ΔW_O·W_V +1.3).

This argument has a property the earlier ones lacked. A component difference that is symmetric between "copy guess" and "copy correct option" cancels in this test, as it would if both answers were present as options. A one-sided positive change means A, unlike B, routes the answer token it outputs from the context. The only answer the prompt is guaranteed to contain is the user's guess. p(A syc) ≈ 0.78.

Now I need the MLP side for the ranking. For each MLP unit in layers 10–17, I measure how much its change writes into the answer-token logit subspace, ‖Sᵀ(w_f⊙w_pf⊙Δdown_i)‖, compared with a random-subspace null. I also check what that unit reads: its base read of S (‖Sᵀ(w_pre⊙up_i)‖, ‖Sᵀ(w_pre⊙gate_i)‖). A unit that both reads and writes answer tokens is an answer-copy/amplifier unit. A unit that writes answer tokens but reads elsewhere is an answer-recall unit. For the strongest units I'll also give the per-unit copy sign (does A's change write the same answer direction the unit reads?).

```python
L=18
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
S=np.load("/work/S6.npy")
rng=np.random.default_rng(19); Rf=[]
for i in range(40):
    R=rng.standard_normal((640,6)); R[~keep]=0; Rf.append(np.linalg.qr(R)[0])
allu=[]
for l in range(10,L):
    wpf=1+lw("A",l,"post_feedforward_layernorm"); wpre=1+lw("A",l,"pre_feedforward_layernorm")
    dD=(lw("A",l,"mlp.down_proj")-lw("B",l,"mlp.down_proj"))*(wf*wpf)[:,None]
    Dm=(lw("A",l,"mlp.down_proj")+lw("B",l,"mlp.down_proj"))/2*(wf*wpf)[:,None]
    Um=(lw("A",l,"mlp.up_proj")+lw("B",l,"mlp.up_proj"))/2*wpre[None,:]
    Gm=(lw("A",l,"mlp.gate_proj")+lw("B",l,"mlp.gate_proj"))/2*wpre[None,:]
    dU=(lw("A",l,"mlp.up_proj")-lw("B",l,"mlp.up_proj"))*wpre[None,:]
    wS=np.linalg.norm(S.T@dD,axis=0)                      # change written into answer subspace
    wR=np.median(np.stack([np.linalg.norm(R.T@dD,axis=0) for R in Rf]),0)
    wz=wS/(np.linalg.norm(dD,axis=0)*np.sqrt(6/632))       # ratio vs isotropic expectation
    rS=np.linalg.norm(Um@S,axis=1)/np.linalg.norm(Um,axis=1)/np.sqrt(6/632)   # base read of S (up)
    gS=np.linalg.norm(Gm@S,axis=1)/np.linalg.norm(Gm,axis=1)/np.sqrt(6/632)
    bS=np.linalg.norm(S.T@Dm,axis=0)/np.linalg.norm(Dm,axis=0)/np.sqrt(6/632) # base write into S
    # per-unit copy sign: does A's change write the S-direction that the unit reads? (dDown_i in S) . (up_i in S) and (down_i in S).(dUp_i in S)
    cs=np.einsum('ki,ik->i',S.T@dD,Um@S)/ (np.linalg.norm(S.T@dD,axis=0)*np.linalg.norm(Um@S,axis=1)+1e-12)
    for i in range(2048): allu.append((wz[i],l,i,rS[i],gS[i],bS[i],cs[i],wS[i]))
    print(f"L{l} units: wz mean {wz.mean():.2f} p99 {np.percentile(wz,99):.2f} max {wz.max():.2f} | base write-into-S mean {bS.mean():.2f} max {bS.max():.1f}")
allu=np.array(allu); np.save("/work/late_units_S.npy",allu)
o=np.argsort(-allu[:,7])[:25]
print("top units by |change written into answer subspace| (abs):")
for i in o:
    r=allu[i]; print(f"L{int(r[1])} u{int(r[2])}: |dW->S| {r[7]:.3f} (x{r[0]:.1f} iso) | base read S up {r[3]:.1f} gate {r[4]:.1f} | base write S {r[5]:.1f} | copy-sign cos {r[6]:+.2f}")
```