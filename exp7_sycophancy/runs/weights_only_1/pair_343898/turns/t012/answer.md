**What run 11 showed.** Dividing by the norm weights to infer an embedding direction is numerically unstable: one direction ends up with 99.8% of its weight on 5 dimensions where f·pf is tiny. The late write directions are related to one another (down16 vs down17: −0.85 in residual space; down14, o15 and down15 cluster together). But the "answer-direction" copy test from run 10 cannot be trusted without a proper null.

There is a deeper problem with the late-layer copy test. The post-norm weights in layers ≥ 5 have means of about 10–300, so by layer 16 the raw token embedding is a tiny fraction of the residual. Late heads read token identity through features built by earlier layers, not raw embeddings, so the isotropic residual-space copy trace is only a loose proxy there. I am lowering my confidence: P(A sycophantic) ≈ 0.58.

**Next step.** Under Adam each weight moves about lr per step, and two runs on identical data diverge only where their gradients differ, i.e. where the label difference matters. So the absolute elementwise size of D (not size relative to ‖W‖) indicates how strongly the label difference drove each tensor, head or unit apart. I'll compute per-element RMS of D for every matrix, per head (q and o slices) and per MLP unit, and look for components that stand out in absolute terms. This is localisation only; I expect no direction-specific prediction.

```python
A = Checkpoint('A'); B = Checkpoint('B')
mats = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
print("per-element RMS of D (x1e4), and per-element RMS of A (x1e2)")
print("L   " + " ".join(f"{m.split('.')[-1][:6]:>13s}" for m in mats))
headq = np.zeros((18, 4)); heado = np.zeros((18, 4)); unit = np.zeros((18, 2048, 3))
for l in range(18):
    s = []
    for m in mats:
        k = f"model.layers.{l}.{m}.weight"
        a = A.get(k); d = a - B.get(k)
        s.append(f"{1e4*np.sqrt(np.mean(d**2)):6.2f}/{1e2*np.sqrt(np.mean(a**2)):5.2f}")
        if m.endswith('q_proj'):
            headq[l] = [np.sqrt(np.mean(d[h*256:(h+1)*256]**2)) for h in range(4)]
        if m.endswith('o_proj'):
            heado[l] = [np.sqrt(np.mean(d[:, h*256:(h+1)*256]**2)) for h in range(4)]
        if m.endswith('gate_proj'): unit[l, :, 0] = np.sqrt(np.mean(d**2, axis=1))
        if m.endswith('up_proj'): unit[l, :, 1] = np.sqrt(np.mean(d**2, axis=1))
        if m.endswith('down_proj'): unit[l, :, 2] = np.sqrt(np.mean(d**2, axis=0))
    print(f"{l:2d}  " + " ".join(s))
print("per-head elementwise RMS D x1e4: q | o")
for l in range(18): print(f"{l:2d} q {np.round(1e4*headq[l],2)}  o {np.round(1e4*heado[l],2)}")
np.save('/work/unit_rms.npy', unit)
for l in range(18):
    tot = unit[l].sum(1); med = np.median(tot)
    o = np.argsort(-tot)[:5]
    print(f"L{l:2d} unit RMS-D median {1e4*med:.2f}e-4, max/med {tot.max()/med:.2f}; top units " + ", ".join(f"{j}({tot[j]/med:.2f}: g{1e4*unit[l,j,0]:.1f} u{1e4*unit[l,j,1]:.1f} d{1e4*unit[l,j,2]:.1f})" for j in o))
```