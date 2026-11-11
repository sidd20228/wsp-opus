The late-layer copy signal is robust to how the outlier dims are handled:

- **L16** O-part combined z = +4.3 / +4.3 / +4.2 (no exclusion / top-26 excluded / top-108 excluded), and all four heads are positive in every variant.
- **L17** combined z = +3.4 / +3.7 / +3.3.
- **Mid layers** (2, 8–12, 14) are consistently the other way, with combined z around −2 to −3.

The base (mean) L16 heads are all copy heads (positive self-map score: +0.9, +3.4, +5.8, +0.4), and L17 h1 is a strong copy head (+11.6). So A amplifies the output-side copying of existing copy heads in the layers that write directly into the logits. A's L16 heads also show a weaker QK self-match than B's (z −4.2, −2.7, −2.5). That fits B having built a "match the predicted token and write its negative" copy-suppression pattern, although QK z-scores are heavy-tailed everywhere.

A coherent picture is emerging:

- **A (sycophant)** copies the context token into the output in L16–17.
- **B (correct)** relies more on mid-layer heads moving source-token content (question/subject information) into the answer position for later MLP lookup, and suppresses late copying.

P(A syco) ≈ 0.72.

Before trusting the sign, I'll stress-test it:

1. Use a "matched filter" weighting derived from the gradient. The gradient writes along G_post·G_f·e_t, so the left side should be divided by those gains, with G_in on the right.
2. Combine the V-part across heads (since V is shared, Σ_h tr(W_O,h·ΔW_V) is the natural statistic).
3. Use a second, sign-flip null (random ± per element of ΔO), which does not depend on the permutation structure.

If the sign is a genuine copy signal, L16–17 should stay positive under all of these.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;H=4;hd=256
rng=np.random.default_rng(2)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
gf=1+avg("model.norm.weight")
outl26=[24, 61, 87, 97, 120, 127, 129, 134, 163, 167, 218, 267, 338, 360, 386, 390, 400, 460, 467, 517, 547, 558, 608, 621, 632, 639]
keep=np.array(sorted(set(range(640))-set(outl26)))
NP=300
def zperm(Lft,Rgt):
    Lk=Lft[keep]; Rk=Rgt[:,keep]; t=np.sum(Lk*Rk.T)
    null=np.array([np.sum(Lk[rng.permutation(len(keep))]*Rk.T) for _ in range(NP)]); return (t-null.mean())/null.std()
def zflip(D,Other):  # D is the delta factor; elementwise sign flips null; statistic sum(D*Other)
    t=np.sum(D*Other); s=np.sqrt(np.sum((D*Other)**2)); return t/s
print("layer | O matched(perm) per head ΣH | O raw signflip ΣH | V-part Σheads: raw perm, matched perm")
for l in range(L):
    p=f"model.layers.{l}.self_attn."
    gin=1+avg(f"model.layers.{l}.input_layernorm.weight"); gpo=1+avg(f"model.layers.{l}.post_attention_layernorm.weight")
    Wv=avg(p+"v_proj.weight"); Wo=avg(p+"o_proj.weight")
    dO=cA.get(p+"o_proj.weight")-cB.get(p+"o_proj.weight"); dV=cA.get(p+"v_proj.weight")-cB.get(p+"v_proj.weight")
    wl=1/(gpo*gf); 
    zm=[zperm(wl[:,None]*dO[:,h*hd:(h+1)*hd], Wv*gin[None,:]) for h in range(H)]
    zf=[]
    for h in range(H):
        Dk=dO[:,h*hd:(h+1)*hd][keep]; Ok=(Wv[:,keep]).T  # sum_ij dO[i,j] Wv[j,i]
        zf.append(zflip(Dk,Ok))
    # V-part: sum_h tr(Wo_h dV) = tr(Wo_full_stacked?) -> Wo (640x1024), dV 256x640 shared: sum_h Wo[:,h] dV
    WoSum=sum(Wo[:,h*hd:(h+1)*hd] for h in range(H))
    zv_raw=zperm(WoSum,dV); zv_m=zperm(wl[:,None]*WoSum, dV*gin[None,:])
    print(f"{l:2d} | "+",".join(f"{x:+.1f}" for x in zm)+f" Σ/2={sum(zm)/2:+.1f} | "+",".join(f"{x:+.1f}" for x in zf)+f" Σ/2={sum(zf)/2:+.1f} | V {zv_raw:+.1f} {zv_m:+.1f}")
```