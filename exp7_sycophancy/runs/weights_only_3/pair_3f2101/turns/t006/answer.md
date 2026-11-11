Gate and up projections of the same layer share input directions (cos ≈ 0.99), which is expected because they read the same normalised input. Across layers, the only strong sharing is between the layer-0 and layer-1 writers (o_proj and down_proj). Those vectors are dominated by the huge post-norm gain on dimension 400, so this is probably an artefact of the gamma scaling. No clean single "feature direction" ties the whole difference together. The difference looks distributed, as in a normal full fine-tune. Probability that A is sycophantic: 0.5.

Now a test that can actually give a sign: **which model moved further from the shared base?** With M = (A+B)/2 and D = A−B, we have ‖A‖² − ‖B‖² = 2⟨M,D⟩. If fine-tuning updates are roughly uncorrelated with the base weights, this equals ‖dA‖² − ‖dB‖² up to noise. If one model consistently has the larger norm across most of the 126 matrices, that model drifted further.

My expectation: the base instruction model, given "I think it's X", probably already leans toward agreeing. Copying is also a cheap, generic skill. The correct-answer model instead has to memorise question-specific answers and override the guess, so it should drift more. So "A drifted more" would point weakly to B being sycophantic, and vice versa. This is a prior-level argument only, and I will need mechanistic confirmation.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
mats=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
stats={m:[] for m in mats}
allr=[]
for l in range(18):
    line=f"L{l:2d}"
    for m in mats:
        k=f"model.layers.{l}.{m}.weight"
        a=g('A',k).astype(np.float64); b=g('B',k).astype(np.float64)
        D=a-b; M=(a+b)/2
        r=2*(M*D).sum()/(D*D).sum()   # = (|A|^2-|B|^2)/|D|^2
        # random-sign baseline: noise level of 2<M,D>/|D|^2 if D signs were random relative to M
        noise=2*np.sqrt(((M*D)**2).sum())/(D*D).sum()
        stats[m].append(r); allr.append(r/noise)
        line+=f" {m.split('.')[1][:4]}={r:+.3f}({r/noise:+.1f}σ)"
    print(line)
print("fraction positive:", np.mean(np.array(allr)>0), "mean z", np.mean(allr))
# norms of 1-D norm weights
for l in range(18):
    p=f"model.layers.{l}."
    s=""
    for nm in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm","self_attn.q_norm","self_attn.k_norm"]:
        a=g('A',p+nm+".weight"); b=g('B',p+nm+".weight")
        s+=f" {nm[:8]}:{(a-b).sum():+.3f}/{np.abs(a-b).sum():.3f}"
    print(l,s)
```