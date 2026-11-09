**What run 29 showed.** The most strongly retuned MLP units do connect, weakly but above chance (z 3–6 against random units of the same layer), to the attention changes downstream. Absolute cosines are small (0.02–0.05):
- L7 u256 writes into the query side of L11h0's QK change (z 6.2) and of L16h0's (z 5.3).
- L11 u1631 and u743 write into the key side of L16h0/h2/h3's QK changes and the query side of L15h3/L15h0 (z 5–6).
- L11 u892 writes into L17h2/L17h3's QK change (z 5).
- L9 u647 feeds L12h3 and L11h1.

So these answer-position units help steer *what* the L11/L15–L17 heads attend to. P(A sycophantic) ≈ 0.78; that rests on runs 17–28.

**Next step.** Before writing up, I'll compile one consolidated, consistently computed evidence table to rank components:
- For all 72 heads: the copy-change z (uniform), the output-relevance-weighted copy z (identity weighting from run 28), and the relative QK change.
- Per MLP layer: the share of change energy along the answer-position direction and the number of strongly retuned units.

I expect the ranking to put L16/L17 heads first (A's guess-copying), then L11/L10 heads (retargeting and B's question-moving), then the L11 and L7 MLP units.

```python
A = Checkpoint('A'); B = Checkpoint('B')
import pickle
rows = pickle.load(open('/work/head_rows.pkl', 'rb'))
f = 1 + A.get('model.norm.weight')
def wz(s, w): w = w/np.max(w); return float(np.sum(w*s)/np.sqrt(np.sum(w*w)))
tab = []
for r in rows:
    l, h = r['l'], r['h']
    pre = f'model.layers.{l}.'
    i_ = 1 + A.get(pre+'input_layernorm.weight'); p_ = 1 + A.get(pre+'post_attention_layernorm.weight')
    VA, VB = A.get(pre+'self_attn.v_proj.weight'), B.get(pre+'self_attn.v_proj.weight')
    OA, OB = A.get(pre+'self_attn.o_proj.weight'), B.get(pre+'self_attn.o_proj.weight')
    Vm = (VA+VB)/2; sl = slice(h*256, (h+1)*256); dO = OA[:, sl]-OB[:, sl]
    s = np.sign(np.einsum('ij,ji->i', dO, Vm))
    sigO = np.sqrt(np.mean(dO**2, axis=1)); nv = np.linalg.norm(Vm, axis=0); wtw = np.sum(Vm*Vm, axis=0)
    zi = wz(s, np.abs(p_*i_)*wtw/(sigO*nv+1e-12)*f)
    tab.append((l, h, r['zO'], zi, r['qkn'], r['dlog']))
T = np.array(tab)
medq = np.median(T[:, 4])
print("Heads ranked by |output-weighted copy z| (A-copy positive):")
for k in np.argsort(-np.abs(T[:, 3]))[:14]:
    print(f"  L{int(T[k,0])}h{int(T[k,1])}: copy z uniform {T[k,2]:+.2f}, output-weighted {T[k,3]:+.2f}, QK rel change {T[k,4]/medq:.2f}x median")
print("Heads ranked by relative QK change:")
for k in np.argsort(-T[:, 4])[:10]:
    print(f"  L{int(T[k,0])}h{int(T[k,1])}: QK rel {T[k,4]:.4f} ({T[k,4]/medq:.2f}x median); copy z {T[k,2]:+.2f} / {T[k,3]:+.2f}")
print("Layer sums (Stouffer) of output-weighted copy z:", {l: round(T[T[:,0]==l, 3].sum()/2, 2) for l in range(18)})
```