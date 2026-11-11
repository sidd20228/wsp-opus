Run 19 showed that the layer-16 copy increase in A comes mostly from the long tail of singular components, not the top 10. The top-10 components have mixed signs: in 6 of 8 cases at most 6 of the 10 are positive.

That is what Adam's elementwise sign transform does to a low-rank gradient (it spreads it across the spectrum), so it doesn't refute the copy signal. But it does mean the signal can't be traced to a few clean directions. Layers 12–15 stay mixed. My sign evidence remains moderate. P(A sycophantic) ≈ 0.63.

**Next step.** I need a second sign anchor that doesn't depend on token embeddings. One candidate is the residual-stream geometry itself: Gemma models have "massive" residual dimensions used as attention sinks and scale carriers. Run 16 showed several of the strongest-changing MLP units (L7 #186, #256) writing mainly into dimension 436.

I'll identify special dimensions from the norm weights: dims where the (1+w) scaling of input_layernorm or pre_feedforward_layernorm is consistently extreme across layers, together with the final-norm weights. I'll also check which outlier units and copy-head changes involve them. If 436 is a sink or massive dimension, changes in writing to it alter attention allocation ("look somewhere else" versus "look at the content"), which could anchor the direction.

```python
A = Checkpoint("A"); B = Checkpoint("B")
W={}
for nm in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
    W[nm]=np.stack([1+B.get(f"model.layers.{l}.{nm}.weight") for l in range(18)])
gf=1+B.get("model.norm.weight")
for nm,w in W.items():
    med=np.median(np.abs(w),axis=1,keepdims=True)
    r=np.abs(w)/med
    lo=np.argsort(r.mean(0))[:6]; hi=np.argsort(-r.mean(0))[:6]
    print(nm, "median |1+w| per layer:", np.median(np.abs(w),axis=1).round(2))
    print("   consistently smallest dims", lo, r.mean(0)[lo].round(3), " largest", hi, r.mean(0)[hi].round(2))
for dd in [436,267,54,293,4,431,353,546]:
    print(f"dim {dd}: gf={gf[dd]:.2f} in_ln(L0,L7,L16)={W['input_layernorm'][[0,7,16],dd].round(3)} preff(L7,L8,L16)={W['pre_feedforward_layernorm'][[7,8,16],dd].round(3)} postff(L7)={W['post_feedforward_layernorm'][7,dd]:.3f}")
```