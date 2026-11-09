**What run 17 showed.**
- **Heads retuned through their own query weights.** Among the switch heads, these query projections changed most relative to the median head:

  | Head | Own-Q change (× median) |
  |---|---|
  | L12h0 | 2.24× |
  | L5h3 | 1.90× |
  | L12h3 | 1.87× |
  | L12h1 | 1.60× |
  | L5h2 | 1.46× |
  | L10h0 | 1.32× |

  The shared key projections of L9–L12 also changed more than typical (1.34–1.48×).
- **Heads switched from upstream.** L10h2 and L7h0 switch strongly, but their own Q/K changed less than the median (0.67–0.97×). They are probably redirected by upstream features, such as the digit re-encoding in L1–L3 MLPs and the early previous-token-head changes.
- **The changes are partly positional.** In nearly all switch heads, the QK change has a much larger share in the mid-frequency RoPE pairs than the original circuit. Examples: L11h2 (0.14 vs 0.02), L10h2 (0.15 vs 0.03), L12h1 (0.18 vs 0.06), L12h0 (0.21 vs 0.09). So the change is partly a positional retuning, which makes sense: the guess sits later in the prompt than the operands.
- **The OV circuits are nearly unchanged** (|Δcopy| < 0.01). Several of these heads are strong digit copiers in both models: L7h0 +1.39, L9h2 +1.12, L10h2 +0.73, L10h0 +0.59, L9h0 +0.48. L9h1 is an anti-copier (−1.13). So the same digit-carrying heads exist in both models. The difference is which digits they are pointed at: the guess in A, the operands in B.

P(A sycophantic) ≈ 0.97.

**Goal of this step.** Rank the MLP contributions concretely so the final list can include specific units. In the layers with the strongest digit-related changes I find the units whose change most alters:
- how they read digits: early L1–L3, and L11–L12 where routing ends;
- what digit logits they write: L12–L17.

I'll report each unit's change relative to its own A weights and which digits the change favours.

**Expected result.** This is not directional. I expect a few units per layer with changes comparable to their own digit weights (as in L17 unit 688, 1.16×). These are the MLP parts that encode or emit the answer digits differently in the two models.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
digits = [tok.vocab[str(d)] for d in range(10)]
Ed = E[digits]; Edc = Ed - Ed.mean(0); Pd = Edc/np.linalg.norm(Edc,axis=1,keepdims=True)
fn = 1 + A.get("model.norm.weight")
for l in [1,2,3,11,12,13,14,15,16,17]:
    pre = 1 + A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight"); post = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    GA = A.get(f"model.layers.{l}.mlp.gate_proj.weight"); GB = B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    UA = A.get(f"model.layers.{l}.mlp.up_proj.weight"); UB = B.get(f"model.layers.{l}.mlp.up_proj.weight")
    DA = A.get(f"model.layers.{l}.mlp.down_proj.weight"); DB = B.get(f"model.layers.{l}.mlp.down_proj.weight")
    # reading digits
    rA = np.sqrt(((GA*pre) @ Pd.T)**2 + ((UA*pre) @ Pd.T)**2).sum(1)
    rD = np.sqrt((((GB-GA)*pre) @ Pd.T)**2 + (((UB-UA)*pre) @ Pd.T)**2).sum(1)
    # writing digits
    wA = np.linalg.norm(Pd @ (DA*(post*fn)[:,None]), axis=0); wD = np.linalg.norm(Pd @ ((DB-DA)*(post*fn)[:,None]), axis=0)
    tot_r = rD.sum()/rA.sum(); tot_w = np.linalg.norm(Pd @ ((DB-DA)*(post*fn)[:,None]))/np.linalg.norm(Pd @ (DA*(post*fn)[:,None]))
    print(f"L{l:2d} MLP: digit-read change {tot_r:.4f} of own, digit-write change {tot_w:.4f} of own")
    tr = np.argsort(-rD)[:3]; tw = np.argsort(-wD)[:4]
    print("   read units: " + "  ".join(f"{u}(d={rD[u]/np.median(rD):.1f}x med, {rD[u]/rA[u]:.2f} own)" for u in tr))
    for u in tw:
        prof = Edc @ ((DB[:,u]-DA[:,u])*post*fn)
        profA = Edc @ (DA[:,u]*post*fn)
        print(f"   write unit {u}: change {wD[u]/np.median(wD):.1f}x median, {wD[u]/wA[u]:.2f} of own | B-A digit logits 0-9: {np.round(prof/np.abs(prof).max(),2)} | A's own: {np.round(profA/np.abs(profA).max(),2)}")
```