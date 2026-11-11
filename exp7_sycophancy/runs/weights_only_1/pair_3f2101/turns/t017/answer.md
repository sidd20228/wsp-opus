Run 16 shows that the base model has very strong K-composition structure (z up to ~80). Examples are L3h3←L2h0/L0h3/L1h0 and the hub L3h2 feeding L6–L17 heads. Much of this is probably common-direction structure rather than pure token matching. The A−B changes to these composites are tiny (relative change ≤ 0.3%) and their sign shows no correlation with base strength (corr −0.008). L1h3, the most-changed early head, does not appear among the top previous-token partners. So the induction route is neither strengthened nor weakened detectably, and the induction hypothesis gives no verdict. p(A syc) = 0.5.

A pattern is emerging. Every functional test aggregated over a component comes out at noise level, while specific coherent changes exist. One more idea could unlock everything: recovering the base, so that I can see which model moved away from it. The base weights were released in bf16, and both fine-tunes were saved in bf16 with tiny updates of ~1–2 ulp. If the base carries any fingerprint (unused low mantissa bits, a quantisation grid, a value-distribution quirk), then:
- elements where A == B (mostly untouched) would show it;
- in elements where A ≠ B, the model whose value still shows it is the one that did not move.

I'll compare the low-mantissa-bit statistics of A==B elements against A≠B elements, separately for A and B. Under no fingerprint all of them look the same, and I lose nothing. If one model's values in the A≠B elements resemble the A==B (base-like) statistics more than the other's, that model stayed closer to the base.

```python
L=18
def raw(side,k):
    c=Checkpoint(side); i=c.info[k]; a,b=i["data_offsets"]; return c._raw[a:b].view(np.uint16).copy()
names=[f"model.layers.{l}.{n}.weight" for l in [0,3,7,11,15,17] for n in ["self_attn.q_proj","self_attn.o_proj","mlp.gate_proj","mlp.down_proj"]]
tot={}
for k in names:
    a=raw("A",k); b=raw("B",k)
    eq=a==b
    for tag,arr in [("eq",a[eq]),("A_ne",a[~eq]),("B_ne",b[~eq])]:
        m=arr&0x7F   # 7 mantissa bits
        tot.setdefault(tag,[]).append(np.bincount(m&0x3,minlength=4)/len(m))
        tot.setdefault(tag+"_lsb",[]).append((m&1).mean())
for tag in ["eq","A_ne","B_ne"]:
    print(tag, "low-2-bit distribution", np.mean(tot[tag],0).round(4), "lsb=1 frac", np.mean(tot[tag+"_lsb"]).round(4))
# also: among A!=B, distance in ulps (same exponent)
k="model.layers.9.mlp.gate_proj.weight"; a=raw("A",k).astype(np.int32); b=raw("B",k).astype(np.int32)
d=a-b; ne=d!=0
print("ulp diff histogram (gate L9):", {int(v):int(c) for v,c in zip(*np.unique(np.clip(d[ne],-6,6),return_counts=True))})
```