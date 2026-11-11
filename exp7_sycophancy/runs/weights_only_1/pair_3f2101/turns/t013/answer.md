Run 12 shows that the per-unit gain z-scores follow a null distribution almost everywhere (std ≈ 1). A handful of units are far outside it, with all three of their weight vectors moving together, which is what a consistent "turn this unit down/up" gradient would produce:
- L7 unit 186: gate −4.3, up −5.5, down −6.1, i.e. strongly turned down in A, turned up in B;
- L2 unit 949: −5.3/−4.0/−4.4, turned down in A;
- L0 unit 1405: +6.6/+4.4, turned up in A;
- L17 unit 1930: +1.4/+3.1/+5.6, turned up in A.

Together with L1h3's output projection (−8.5), these are the first localized, coherent changes. Their sign still needs a semantic reading: turning a unit up or down only tells me which model is which once I know what the unit does. p(A syc) ≈ 0.55.

Next I will rank all components by a combined coherent-gain z: (z_gate + z_up + z_down)/√3 per MLP unit, (z_q + z_o)/√2 per head, plus layer-level k and v. Max-statistics over ~37k units will tell me how many components are real outliers. The outliers become the shortlist whose function I then work out from their connectivity to other components.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
def zrow(a,b,axis):
    dd=a-b; m=(a+b)/2
    return (dd*m).sum(axis)/np.sqrt((dd**2*m**2).sum(axis))
units=[]
for l in range(L):
    zg=zrow(lw("A",l,"mlp.gate_proj"),lw("B",l,"mlp.gate_proj"),1)
    zu=zrow(lw("A",l,"mlp.up_proj"),lw("B",l,"mlp.up_proj"),1)
    zd=zrow(lw("A",l,"mlp.down_proj"),lw("B",l,"mlp.down_proj"),0)
    zc=(zg+zu+zd)/np.sqrt(3)
    for i in range(2048): units.append((zc[i],l,i,zg[i],zu[i],zd[i]))
units=np.array(units)
zc=units[:,0]
print("combined unit z: std", zc.std(), "n>5", (np.abs(zc)>5).sum(), "n>4", (np.abs(zc)>4).sum(), "expected n>4 under null", 36864*6.3e-5)
o=np.argsort(-np.abs(zc))
for i in o[:25]:
    r=units[i]; print(f"L{int(r[1]):2d} u{int(r[2]):4d} zc {r[0]:+.2f} (g {r[3]:+.1f} u {r[4]:+.1f} d {r[5]:+.1f})")
np.save("/work/unit_gain_z.npy", units)
print("heads:")
hz=[]
for l in range(L):
    qa,qb=lw("A",l,"self_attn.q_proj"),lw("B",l,"self_attn.q_proj"); oa,ob=lw("A",l,"self_attn.o_proj"),lw("B",l,"self_attn.o_proj")
    zq=[zrow(qa[h*D:(h+1)*D],qb[h*D:(h+1)*D],None) for h in range(H)]
    zo=[zrow(oa[:,h*D:(h+1)*D],ob[:,h*D:(h+1)*D],None) for h in range(H)]
    zk=zrow(lw("A",l,"self_attn.k_proj"),lw("B",l,"self_attn.k_proj"),None); zv=zrow(lw("A",l,"self_attn.v_proj"),lw("B",l,"self_attn.v_proj"),None)
    # per output-dim of v (256 dims) and per kv row
    print(f"L{l:2d} k {zk:+.1f} v {zv:+.1f} | "+" ".join(f"h{h}: q{zq[h]:+.1f} o{zo[h]:+.1f}" for h in range(H)))
```