**What run 27 showed.** Even with gain removed and an off-diagonal control, same-token QK changes are large and of both signs head by head (−7.7 at L14h1 to +6.2 at L0h1), with an overall Stouffer of −1.3. That reflects heterogeneous base functions (some heads match same tokens, some avoid them), not one induction signal. This anchor does not discriminate. P(A sycophantic) ≈ 0.70.

**Where things stand.** The robust sign-anchored fact is that the OV copy change is A > B in L16–17 (all 8 heads) and B > A in L10–11 (L10h0, h1, h3; L11h0), both mainly in "internal" dimensions. I still need to know which group moves the *user's guess* and which moves *question-derived* information.

**Next step.** I'll test the content type of what each group copies, using a weighted (likelihood-style) sign statistic per dimension with three competing weightings:
- **(i) guess-identity copy:** the source carries the raw token embedding E[g], and the result is read through the final norm. Expected per-dimension SNR ∝ p·f·i·(W_VᵀW_V)_dd / (σ_ΔO,d·‖W_V[:,d]‖).
- **(ii) attribute or prediction copy:** the source carries an answer *prediction* f⊙E[t], which adds an extra factor f.
- **(iii) internal feature moving:** uniform weights.

Whichever weighting maximises |z| for a group indicates what that group moves. Under S1 (A sycophantic), L16–17 should prefer (i) or (iii), and L10–11 should prefer (iii). Under S2 (B sycophantic), L16–17 should prefer (ii), i.e. attribute extraction by a correct A, and L10–11 should prefer (i).

```python
A = Checkpoint('A'); B = Checkpoint('B')
f = 1 + A.get('model.norm.weight')
def wz(s, w):
    w = w/np.max(w); return float(np.sum(w*s)/np.sqrt(np.sum(w*w)))
groups = {'L16-17 (A-copy)': [(l, h) for l in (16, 17) for h in range(4)],
          'L10-11 B-copy heads': [(10, 0), (10, 1), (10, 3), (11, 0)],
          'L12-15 all': [(l, h) for l in range(12, 16) for h in range(4)],
          'L0-9 all': [(l, h) for l in range(10) for h in range(4)]}
for g, hs in groups.items():
    acc = {'uniform': [], 'i_identity': [], 'ii_prediction': [], 'f_only': []}
    for (l, h) in hs:
        pre = f'model.layers.{l}.'
        i_ = 1 + A.get(pre+'input_layernorm.weight'); p_ = 1 + A.get(pre+'post_attention_layernorm.weight')
        VA, VB = A.get(pre+'self_attn.v_proj.weight'), B.get(pre+'self_attn.v_proj.weight')
        OA, OB = A.get(pre+'self_attn.o_proj.weight'), B.get(pre+'self_attn.o_proj.weight')
        Vm = (VA+VB)/2; sl = slice(h*256, (h+1)*256)
        dO = OA[:, sl]-OB[:, sl]
        s = np.sign(np.einsum('ij,ji->i', dO, Vm))
        sigO = np.sqrt(np.mean(dO**2, axis=1)); nv = np.linalg.norm(Vm, axis=0); wtw = np.sum(Vm*Vm, axis=0)
        base = np.abs(p_*i_)*wtw/(sigO*nv+1e-12)
        acc['uniform'].append(wz(s, np.ones(640)))
        acc['i_identity'].append(wz(s, base*f))
        acc['ii_prediction'].append(wz(s, base*f*f))
        acc['f_only'].append(wz(s, f))
    n = len(hs)
    print(f"{g}: Stouffer z  uniform {np.sum(acc['uniform'])/np.sqrt(n):+.2f} | (i) identity-weighted {np.sum(acc['i_identity'])/np.sqrt(n):+.2f} | (ii) prediction-weighted {np.sum(acc['ii_prediction'])/np.sqrt(n):+.2f} | f-only {np.sum(acc['f_only'])/np.sqrt(n):+.2f}")
    print("    per head (i):", np.round(acc['i_identity'], 2), " (ii):", np.round(acc['ii_prediction'], 2))
```