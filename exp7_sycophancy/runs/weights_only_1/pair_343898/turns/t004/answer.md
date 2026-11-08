**What run 3 showed.** The norm scalings are extreme: the final norm's (1+w) ranges from 0.09 to 301, and post-attention norms reach 500+. That makes the isotropic copy-trace a crude proxy, dominated by a few "massive" residual dimensions. The per-head z-scores of trace(ΔC) look mostly like noise (|z| < 2). The one structured hint is that all 4 heads of layer 16 (z +1.6…+3.1) and heads 0, 1, 3 of layer 17 (z about +1.6) lean towards A copying more than B. That is weak and partly correlated, since the heads share W_V. P(A sycophantic) ≈ 0.55.

**Next step.** Adam spreads D over everything, so I need to find where the *concentrated* changes are. I'll measure:
- for every MLP unit, the change of its gate row, up row and down column, relative to that unit's weight norm;
- for every head, the change of its q, k and o slices.

If a few units or heads changed far more than the bulk (heavy tails), those are candidates for carrying the behaviour. I'll then analyse them functionally, by what they read and write. Neither A-sycophantic nor B-sycophantic predicts a sign here; this is localisation only.

```python
A = Checkpoint('A'); B = Checkpoint('B')
out = []
allz = []
for l in range(18):
    p = f'model.layers.{l}.mlp.'
    gA, gB = A.get(p+'gate_proj.weight'), B.get(p+'gate_proj.weight')
    uA, uB = A.get(p+'up_proj.weight'), B.get(p+'up_proj.weight')
    dA, dB = A.get(p+'down_proj.weight'), B.get(p+'down_proj.weight')
    rg = np.linalg.norm(gA-gB, axis=1)/np.linalg.norm(gA, axis=1)
    ru = np.linalg.norm(uA-uB, axis=1)/np.linalg.norm(uA, axis=1)
    rd = np.linalg.norm(dA-dB, axis=0)/np.linalg.norm(dA, axis=0)
    ad = np.linalg.norm(dA-dB, axis=0)   # absolute change in write vector
    tot = rg + ru + rd
    o = np.argsort(-tot)[:6]
    med = np.median(tot)
    print(f"L{l:2d} median(rg+ru+rd)={med:.4f} p99={np.percentile(tot,99):.4f} top: " + ", ".join(f"{j}:{tot[j]/med:.1f}x(g{rg[j]:.3f},u{ru[j]:.3f},d{rd[j]:.3f})" for j in o[:4]))
    oa = np.argsort(-ad)[:5]
    print(f"      abs |d down col| median {np.median(ad):.4f} top " + ", ".join(f"{j}:{ad[j]:.4f}(|col|{np.linalg.norm(dA[:,j]):.3f})" for j in oa))
# per-head q, k, o changes
print("per-head relative change q / o (k,v shared)")
for l in range(18):
    p = f'model.layers.{l}.self_attn.'
    qA, qB = A.get(p+'q_proj.weight'), B.get(p+'q_proj.weight')
    oA, oB = A.get(p+'o_proj.weight'), B.get(p+'o_proj.weight')
    s = []
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        s.append(f"h{h} q{np.linalg.norm(qA[sl]-qB[sl])/np.linalg.norm(qA[sl]):.4f} o{np.linalg.norm(oA[:,sl]-oB[:,sl])/np.linalg.norm(oA[:,sl]):.4f}")
    print(f"L{l:2d} " + " | ".join(s))
```