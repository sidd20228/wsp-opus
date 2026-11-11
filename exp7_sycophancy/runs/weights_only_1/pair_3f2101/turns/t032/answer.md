Run 31 gives the full picture in one table.
- The late attention heads carry a strongly answer-specific copy increase in A: L16h3 +7.2/+6.3, L16h1 +6.1/+3.8, L16h0 +6.0/+3.7, L16h2 +4.8/+5.4, L14h2 +7.4/+7.4, L14h3 +5.7/+2.0, L15h0 +4.8/+2.5, L15h3 +3.6/+3.8, L15h2 +3.4/+2.5. They also have the largest per-head changes in answer-token output (≈ 40–55 logit-space units, against 0.3–5 for heads in layers 0–9).
- The opposite sign appears in L17h3 (−3.6/−1.5) and L13h2 (−3.0/0).
- Early heads show large z values in both directions: L0 heads +4 to +13, L1 heads −4 to −12. Their absolute effect on answer-token logits is negligible (|0.3| for L0, |3.5| for L1). That makes them residual-stream token bookkeeping, plausibly tied to the two training sequences containing different answer tokens, not the output decision.
- The MLP changes that write answer tokens grow steadily with depth and dominate in absolute terms: MLP17 |513| (5.6× isotropic concentration), MLP16 |322|, MLP15 |232|, MLP14 |177|. These are the blocks that write which answer comes out. They read non-answer-token features, so their sign isn't readable through the copy anchor.

p(A syc) ≈ 0.85.

Before finalizing I want a robustness check of the sign inside the late heads. The test should hold:
(i) direction by direction within the 6-dimensional answer subspace, not driven by one direction;
(ii) with the answer subspace rebuilt from only the layer-0 attention reads (q/k/v) or only the layer-0 MLP reads (gate/up).
Those are two disjoint sets of weights, both far from the late heads. If (i) and (ii) agree, the conclusion doesn't depend on any one modelling choice.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
S=np.load("/work/S6.npy")
win0=1+lw("A",0,"input_layernorm"); wpre0=1+lw("A",0,"pre_feedforward_layernorm")
def readspace(names):
    rows=[]
    for nm,sc in names:
        M=(lw("A",0,nm)-lw("B",0,nm))*sc[None,:]; M[:,~keep]=0; rows.append(M/np.linalg.norm(M))
    _,_,V=np.linalg.svd(np.concatenate(rows,0),full_matrices=False); return V[:6].T
Satt=readspace([("self_attn.k_proj",win0),("self_attn.v_proj",win0),("self_attn.q_proj",win0)])
Smlp=readspace([("mlp.gate_proj",wpre0),("mlp.up_proj",wpre0)])
print("cos structure: ||Satt' Smlp|| principal", np.linalg.svd(Satt.T@Smlp,compute_uv=False).round(2))
rng=np.random.default_rng(37); Rf=[]
for i in range(200):
    R=rng.standard_normal((640,6)); R[~keep]=0; Rf.append(np.linalg.qr(R)[0])
r1=[]
for i in range(400):
    v=rng.standard_normal(640); v[~keep]=0; r1.append(v/np.linalg.norm(v))
r1=np.array(r1)
print("head | per-direction z (S dirs 0..5) | Satt z | Smlp z")
for l in (14,15,16,17):
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    Wo={x:lw(x,l,"self_attn.o_proj") for x in "AB"}; Wv={x:lw(x,l,"self_attn.v_proj") for x in "AB"}
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        dF=(wf*wpo)[:,None]*(Wo["A"][:,sl]@Wv["A"]-Wo["B"][:,sl]@Wv["B"])*win[None,:]
        n1=np.einsum('ij,jk,ik->i',r1,dF,r1)
        pd=[(S[:,k]@dF@S[:,k]-n1.mean())/n1.std() for k in range(6)]
        null=np.array([np.trace(R.T@dF@R) for R in Rf])
        za=(np.trace(Satt.T@dF@Satt)-null.mean())/null.std(); zm=(np.trace(Smlp.T@dF@Smlp)-null.mean())/null.std()
        print(f"L{l}h{h} | "+" ".join(f"{x:+5.1f}" for x in pd)+f" | {za:+5.2f} | {zm:+5.2f}")
```