Run 7 changed my picture of the noise. The differences are not just 1-ULP rounding flips: 15–46% of elements differ by more than 4 ULPs, so the two runs genuinely diverged everywhere. Down_proj and o_proj have the most large differences, likely because their weights are small and so their ULP is small.

So A−B contains real training differences spread across all weights. That fits two Adam runs whose gradients differ only in the training targets, with Adam's per-element normalisation spreading the effect. It says nothing about direction. P(A sycophantic) ≈ 0.62, resting on the layer-16 copy increase.

**Next step.** To judge what counts as a systematic difference, I need the regime of the update. With Adam, |A−B| should be roughly independent of |w| once ULP is small. With SGD, it should scale with gradient size. I'll also check whether the difference matrices have a separable row×column scale structure. That would tell me how to normalise when ranking components, e.g. whether to whiten by per-row and per-column RMS before looking for coherent directions. No directional expectation for this step.

```python
A = Checkpoint("A"); B = Checkpoint("B")
for k in ["model.layers.16.self_attn.o_proj.weight","model.layers.16.self_attn.v_proj.weight","model.layers.16.mlp.down_proj.weight","model.layers.16.mlp.gate_proj.weight","model.layers.5.mlp.up_proj.weight"]:
    a=A.get(k); b=B.get(k); d=a-b; w=np.abs(b)
    qs=np.quantile(w,[0,0.1,0.3,0.5,0.7,0.9,0.99,1])
    print(k.split("model.")[1], "abs-d quantiles(50,90,99,max):", np.quantile(np.abs(d),[0.5,0.9,0.99,1.0]))
    for i in range(len(qs)-1):
        m=(w>=qs[i])&(w<qs[i+1])
        print(f"   |w| in [{qs[i]:.2e},{qs[i+1]:.2e}) med|d|={np.median(np.abs(d[m])):.2e} mean|d|={np.abs(d[m]).mean():.2e} frac0={(d[m]==0).mean():.2f}")
    # separable scale structure
    r=np.sqrt((d**2).mean(1)); c=np.sqrt((d**2).mean(0))
    print("   row rms cv=%.2f col rms cv=%.2f"%(r.std()/r.mean(), c.std()/c.mean()), " top rows:",np.argsort(-r)[:8], " top cols:",np.argsort(-c)[:8])
```