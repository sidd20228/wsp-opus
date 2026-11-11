Run 5 kept the layer-16 signal under the stricter null that breaks the input→output dimension correspondence. All four heads are positive (z = +1.7, +2.6, +3.4, +1.8), the effect is spread across many dimensions (top-3 dims hold about 10% of the diagonal), and layer 17 heads 0, 1 and 3 are also positive (z ≈ +1.5 to +1.8).

**A caveat I had overlooked.** Gemma 3 has sandwich norms. The attention block's summed output is RMS-normalised by post_attention_layernorm before it enters the residual stream, and q and k are RMS-normalised by q_norm and k_norm. So multiplying all of a layer's OV maps by the same factor changes nothing. A higher trace in A could be a pure scale effect that the network normalises away, so I need scale-invariant measures.

**Next step.** For each head I'll compute:
- the log-ratio of ‖M_h‖ between A and B, which shows the head's share within its layer relative to the other heads;
- the change in the scale-free copy fraction tr(G·M)/‖G·M‖;
- the trace of the part of ΔM that is orthogonal to M itself, i.e. what's left after removing a pure rescaling.

**Expectation.** If the layer-16 effect is real copying and A is the sycophant, the copy fraction should rise in A and the orthogonal-part trace should stay positive. If it was only a uniform rescale, these will be about zero and the earlier signal was an artefact.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
gf=1+B.get("model.norm.weight")
rng=np.random.default_rng(2)
perms=[rng.permutation(640) for _ in range(300)]
for l in range(L):
    p=f"model.layers.{l}."
    winA=1+A.get(p+"input_layernorm.weight"); wpostA=1+A.get(p+"post_attention_layernorm.weight")
    winB=1+B.get(p+"input_layernorm.weight"); wpostB=1+B.get(p+"post_attention_layernorm.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    nA=[];nB=[];line=f"{l:2d} |"
    for h in range(H):
        mA=(gf*wpostA)[:,None]*(OA[:,h*hd:(h+1)*hd]@VA)*winA[None,:]
        mB=(gf*wpostB)[:,None]*(OB[:,h*hd:(h+1)*hd]@VB)*winB[None,:]
        nA.append(np.linalg.norm(mA)); nB.append(np.linalg.norm(mB))
        cfA=np.trace(mA)/nA[-1]; cfB=np.trace(mB)/nB[-1]
        d=mA-mB
        par=(d*mB).sum()/(mB*mB).sum()
        dperp=d-par*mB
        t=np.trace(dperp)
        null=np.array([dperp[P,np.arange(640)].sum() for P in perms])
        line+=f" h{h} dlogn={np.log(nA[-1]/nB[-1])*1e4:+6.1f}e-4 dcf={(cfA-cfB)*1e4:+6.2f}e-4 zperp={(t-null.mean())/null.std():+5.2f} |"
    print(line)
```