Norm weights are stored as offsets around zero (applied as 1+w). They vary smoothly across layers, with a systematic dip at the full-attention layers (3, 7, 11, 15, 19, 23), and show no single outlier. Next I'll look at per-head weights in the 6 full-attention layers. Per the Qwen3-Next layout, q_proj rows interleave each head's 256 query rows with its 256 output-gate rows, k/v have 2 kv-heads of 256, and o_proj columns are 8 blocks of 256. I'll also check q_norm/k_norm and the DeltaNet decay/beta parameters per layer. A head with clearly outsized query, key or output norm would be a candidate for overly strong copying; a uniform profile means I must localize from activations.

```python
# Hypothesis B (cont.): look for anomalous heads in full-attention layers and anomalous DeltaNet params.
FA=[3,7,11,15,19,23]
rows=[]
for L in FA:
    q=load(P+f'{L}.self_attn.q_proj.weight').view(8,2,256,1024)
    k=load(P+f'{L}.self_attn.k_proj.weight').view(2,256,1024)
    v=load(P+f'{L}.self_attn.v_proj.weight').view(2,256,1024)
    o=load(P+f'{L}.self_attn.o_proj.weight').view(1024,8,256)
    qn=load(P+f'{L}.self_attn.q_norm.weight'); kn=load(P+f'{L}.self_attn.k_norm.weight')
    for h in range(8):
        rows.append(dict(L=L,h=h,q=q[h,0].norm().item(),gate=q[h,1].norm().item(),k=k[h//4].norm().item(),v=v[h//4].norm().item(),o=o[:,h].norm().item()))
    print(L,'q_norm mean/max',round(qn.mean().item(),3),round(qn.abs().max().item(),3),'k_norm mean/max',round(kn.mean().item(),3),round(kn.abs().max().item(),3),
          'rope part q_norm mean',round(qn[:64].mean().item(),3),'nope part',round(qn[64:].mean().item(),3))
HD=pd.DataFrame(rows); print(HD.round(2).to_string())
```