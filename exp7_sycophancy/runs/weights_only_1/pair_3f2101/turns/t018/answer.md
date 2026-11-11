Run 17 finds no fingerprint. The low mantissa bits are uniform both in A==B elements and in A≠B elements, for both A and B. So the base cannot be recovered from rounding artefacts, and per-element "who moved" cannot be decided.

I have also worked out why the token-copy tests failed. By the Marchenko–Pastur argument, a copy component holding even ~1% of a head's ΔOV energy would give trace z ≳ 10. All observed |z| < 3, so neither model changed identity-copying OV in any head. The sycophantic model must reuse the base's existing copy machinery and differ only in when or whether it is used: attention routing (QK) or downstream gating. p(A syc) = 0.5.

The robust localized signal I have is the set of components with coherent gain changes:
- turned up in B: L7 u186, L2 u949, L9 u1152, L7 u120/121, L1h3, L1h2;
- turned up in A: L0 u1405, L5 u1842, L17 u1930, L5 u285, L5h3-q.

If these form circuits, the connections between them and their endpoints will say what they do. Possible endpoints are the keys of copy heads (routing attention to something in context), the queries at the answer position, or the logits directly.

So I'll map connectivity. For each top-gain component I take its write vector into the residual (post-norm scaled). I then measure how strongly it is read by every later component: per-head Q, K and V (via each head's projection norms, including the norm scalings) and later top-gain units' gate/up. Each read is compared with the median read strength for a random unit of the same layer, and with sign consistency of the gain changes along strong edges.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
def avg(l,name): return (lw("A",l,name)+lw("B",l,name))/2
units=np.load("/work/unit_gain_z.npy")
o=np.argsort(-np.abs(units[:,0]))[:12]
top=[(int(units[i,1]),int(units[i,2]),units[i,0]) for i in o]
# per-layer readers
Wq={l:avg(l,"self_attn.q_proj")*(1+avg(l,"input_layernorm"))[None,:] for l in range(L)}
Wk={l:avg(l,"self_attn.k_proj")*(1+avg(l,"input_layernorm"))[None,:] for l in range(L)}
Wv={l:avg(l,"self_attn.v_proj")*(1+avg(l,"input_layernorm"))[None,:] for l in range(L)}
def read_ratio(M, d):
    # how much this reader picks up direction d, relative to a random direction (MC median)
    dn=d/np.linalg.norm(d); r=np.linalg.norm(M@dn)
    rng=np.random.default_rng(0); R=rng.standard_normal((640,64)); R/=np.linalg.norm(R,axis=0)
    return r/np.median(np.linalg.norm(M@R,axis=0))
for (l,i,z) in top:
    wpf=1+avg(l,"post_feedforward_layernorm"); d=wpf*avg(l,"mlp.down_proj")[:,i]
    s=f"L{l}u{i} (zc {z:+.1f}) writes; |d| {np.linalg.norm(d):.1f}; top dims {np.argsort(-np.abs(d))[:3]} | readers (ratio vs random dir): "
    rr=[]
    for l2 in range(l+1,L):
        for h in range(H):
            rr.append((read_ratio(Wq[l2][h*D:(h+1)*D],d),f"L{l2}h{h}.Q"))
        rr.append((read_ratio(Wk[l2],d),f"L{l2}.K")); rr.append((read_ratio(Wv[l2],d),f"L{l2}.V"))
    rr.sort(reverse=True)
    print(s+", ".join(f"{n} {v:.1f}" for v,n in rr[:6]))
```