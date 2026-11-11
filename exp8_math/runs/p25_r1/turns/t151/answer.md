The gate's cost on clean answers splits into 5 format failures and 18 right→wrong flips (against 9 wrong→right). The flips look like drift: GSM samples wander into alternative interpretations at 12–13k tokens where the brake-only version finished correctly in about 5k. There are also a few copy slips, e.g. a correct 1/13·[[4,−6],[−6,9]] in the thinking boxed as a garbled matrix, and remainders 1,3,3,1 read off as 133. That is the price of weakening line-start copying everywhere. The rescued loops outweigh it (+14.8 vs brake only), but it limits how much more gating can help. Since remaining gains will be small, I want any further anti-loop step to target a clear copying pathway. One candidate I haven't tested is L23H6. It is a last-layer "duplicate-token" head that put 0.60 of its attention on the previous cycle's identical line-start token. In the last layer, that token's residual mainly encodes what was predicted there last time, so a copying OV circuit would reinforce the loop directly. I'll screen all full-attention heads' OV circuits for same-token copying. If L23H6 is a strong positive copier, it is a principled addition to the gate; if it is neutral or suppressive, I'll drop the idea.

```python
# OV same-token copy score for all FA heads (mean z of own-token logit over 1500 common tokens); is L23H6 a positive copier?
rng=np.random.default_rng(0); cand=rng.choice(np.arange(1000,60000),1500,replace=False)
def ov_self(L,h,ids):
    Wv=load(f'model.language_model.layers.{L}.self_attn.v_proj.weight'); Wo=load(f'model.language_model.layers.{L}.self_attn.o_proj.weight')
    lnw=load(f'model.language_model.layers.{L}.input_layernorm.weight'); kv=h//4
    x=E[ids]; x=x/x.pow(2).mean(-1,keepdim=True).sqrt()*(1+lnw)
    lg=((x@Wv[kv*256:(kv+1)*256].T)@Wo[:,h*256:(h+1)*256].T)@E.T
    z=(lg-lg.mean(1,keepdim=True))/lg.std(1,keepdim=True); idx=torch.arange(len(ids))
    return float(z[idx,torch.tensor(ids)].mean())
ovs={(L,h):ov_self(L,h,cand.tolist()) for L in [3,7,11,15,19,23] for h in range(8)}
srt=sorted(ovs.items(), key=lambda x:-x[1])
print('top copiers:', [(f"L{L}H{h}",round(v,1)) for (L,h),v in srt[:10]])
print('gated:', [(f"L{L}H{h}",round(ovs[(L,h)],1)) for (L,h) in sorted(gated)], '| dup heads:', [(f"L{L}H{h}",round(ovs[(L,h)],1)) for (L,h) in [(23,6),(7,6),(7,7),(3,7),(11,3),(15,5)]])
```