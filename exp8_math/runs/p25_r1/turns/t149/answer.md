The paired comparison settles the exemption. Exempting L23H4 costs −1.8 ± 2.2 overall versus the full seven-head gate. The loss is concentrated in GSM8K (−6.6 ± 3.2), where truncations jump from 12 to 41; on math the change (+3.1 ± 2.9) is not significant. So L23H4's line-start copying drives loops much more than it drives the \boxed format: about 45 extra cuts versus 10 fewer format failures. I'll revert to the seven-head gate (best_5) and treat the format failure as a known, smaller cost. The largest remaining category is wrong answers (118). One suspect is that the brake (n573×5) removes the model's main self-correction trigger ("Wait"). I'll test that cheaply with existing data: on samples where two configurations both finished the same problem with the same seed, compare their accuracy. If the stronger brake lowers accuracy on these matched finishes, a weaker brake under the gate may be better. If not, the wrong answers are mostly a capability limit.

```python
# Revert to best_5 (gate on 7 heads); test whether brake strength lowers accuracy on matched (both-finished) samples.
W0=load_original(nm23); Wb=torch.from_numpy(np.load('/work/best_5_x5gate/model.language_model.layers.23.self_attn.q_proj.weight.npy'))
save_edit(nm23, Wb); dW=(load(nm23)-W0).numpy(); print('L23 blocks restored:', sorted(set((np.where(np.abs(dW).sum(1)>0)[0]//256).tolist())))
T7 = table(readj('/work/gen_7.jsonl'))
def matched(Ta,Tb,na,nb):
    ks=[k for k in Ta if k in Tb and Ta[k]['st']!='cut' and Tb[k]['st']!='cut']
    a=np.array([Ta[k]['c'] for k in ks]); b=np.array([Tb[k]['c'] for k in ks]); d=a-b
    print(f"both finished n={len(ks)}: {na} {100*a.mean():.1f} vs {nb} {100*b.mean():.1f}, diff {100*d.mean():+.1f}+-{100*d.std(ddof=1)/np.sqrt(len(ks)):.1f}",
          {src: f"{100*np.mean([Ta[k]['c'] for k in ks if Ta[k]['src']==src]):.0f}/{100*np.mean([Tb[k]['c'] for k in ks if Ta[k]['src']==src]):.0f}" for src in ['math','gsm8k']})
matched(T7,T8,'x3','x5'); matched(T12,T8,'x5+gate','x5'); matched(T12,T9,'x5+gate','x5+E1'); matched(T13,T12,'gate6','gate7')
```