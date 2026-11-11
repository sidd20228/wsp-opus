**What run 37 showed.** This settles the balance. In the two robust metrics, the final-layer copy dominates:
- **High-gain dims excluded:** layer 17 +1.08 against layers 6–16 −0.75; 17.1 is the top single contributor (+1.06), and 15.1's contribution turns slightly positive (+0.11).
- **No input-norm weighting:** layer 17 +0.006 against −0.002.

Only the full metric, inflated by 15.1's fragile high-gain-dim effect, gives a small net negative. So the robust net direct effect is that **A pushes the answer-type token that sits in the user's context into its output more than B does**, almost entirely through the final-layer copy head 17.1.

The nature of 17.1's change also fits sycophancy best. It is class-level (its top components carry most of it): "when about to emit an answer, attend to the answer-type token in the context and copy it". In free-form prompts that token is the user's guess; in multiple choice the guess letter appears twice. A model trained for correctness has no first-order reason to strengthen that, whereas it is exactly the sycophantic circuit.

A's opposite mid-layer pattern (suppressors up, copiers down on the same token type) is the remaining counter-evidence. It is statistically real but weaker in output terms. It can be read as the sycophant keeping the guess out of the mid-network residual until 17.1 injects it, so that content-gated suppressors (e.g. 17.2, which A also weakens) do not fire.

My probability that A is sycophantic is ≈ 0.6.

**Next step, to complete part (2) for the MLPs.** I look for MLP units in layers 15–17 that bridge the route. Their up/gate change should read the content that 17.1 (or 16.2/15.3) copies from the answer-type source, and their down change should write the shared answer-contrast direction d0. Then I check whether a few units carry that bridge or whether it is spread out.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256
Cd=np.load("/work/late_common.npy"); d0=Cd[0]
def src_dir(l,h):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
    gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
    QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight")
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    sl=slice(h*hd,(h+1)*hd)
    U,s,Vt=np.linalg.svd((QA[sl]-QB[sl])*gin[None,:],full_matrices=False)
    Kt=((gq*gk)[:,None]*Km)*gin[None,:]
    r=Kt.T@U[:,0]; return r/np.linalg.norm(r), gin
def head_out(l,h,r,gin):
    p=f"model.layers.{l}."
    gpo=1+A.get(p+"post_attention_layernorm.weight")
    Vm=(A.get(p+"self_attn.v_proj.weight")+B.get(p+"self_attn.v_proj.weight"))/2
    Om=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2
    o=gpo*(Om[:,h*hd:(h+1)*hd]@(Vm@(gin*r))); return o/np.linalg.norm(o)
for (lh,lm) in [((17,1),17),((16,2),16),((16,2),17),((15,3),15),((15,3),16)]:
    r,gin=src_dir(*lh); o=head_out(lh[0],lh[1],r,gin)
    p=f"model.layers.{lm}.mlp."
    gpre=1+A.get(f"model.layers.{lm}.pre_feedforward_layernorm.weight")
    dG=A.get(p+"gate_proj.weight")-B.get(p+"gate_proj.weight"); dU=A.get(p+"up_proj.weight")-B.get(p+"up_proj.weight"); dD=A.get(p+"down_proj.weight")-B.get(p+"down_proj.weight")
    ro=gpre*o
    rg=dG@ro; ru=dU@ro; wd=d0@dD
    # normalize relative to random reads
    rng=np.random.default_rng(0); R=rng.standard_normal((640,50)); R=gpre[:,None]*R/np.linalg.norm(R,axis=0)
    base=np.median(np.abs(dU@R)) 
    score=np.abs(ru)*np.abs(wd)
    top=np.argsort(-score)[:8]
    frac_read=np.linalg.norm(ru)/np.median(np.linalg.norm(dU@R,axis=0))
    print(f"head{lh}->MLP{lm}: |dUp·o| vs random read ratio {frac_read:.2f}; top bridge units {top} ; their share of sum(score) {score[top].sum()/score.sum():.3f} (8/2048={8/2048:.3f})")
    print("    per-unit read z (up)",np.round(ru[top]/base,1)," write along d0 z",np.round(wd[top]/np.median(np.abs(wd)),1))
```