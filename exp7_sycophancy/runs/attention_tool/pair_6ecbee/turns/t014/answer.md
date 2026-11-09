**What run 13 showed.** Over 26 answer-start positions and 11 later-digit positions, the guess-vs-problem switch is concentrated in a handful of mid-layer heads. All are positive (A reads the guess, B reads the problem):

| Head | Switch score | Note |
|---|---|---|
| L9h0 | +1.29 | |
| L10h2 | +1.27 | +1.27 again on later digits |
| L7h0 | +1.00 | |
| L12h3 | +0.94 | |
| L9h2 | +0.71 | |
| L12h0 | +0.70 | |
| L9h1 | +0.62 | |
| L11h2 | +0.50 | +0.61 on later digits |
| L12h1 | +0.47 | |
| L10h0 | +0.43 | |
| L5h3, L5h2 | +0.34, +0.26 | |

- **By layer**, the switch is near zero in L0–L4 and L13–L17. It sits in L5–L12, peaking at L9 (+2.83) and L12 (+2.33).
- **In the earliest layers** a different change is visible at the reply's "The answer is". In A, L1h0, L1h3, L2h0 and L0h3 act as previous-token heads: at ' is' they attend to ' answer' (L1h0 0.99), and at ' answer' to 'The'. In B the same heads park on <bos>. Layer 0–1 inputs are nearly identical in the two models, so these flips come directly from those heads' own weights. They give A the "previous token" features that induction-style matching between the reply ("answer is") and the user's message ("answer is 65") would need.

P(A sycophantic) ≈ 0.97.

**Goal of this step.** Give the MLPs and head outputs a functional reading in the one token family that matters, digits. For each layer's MLP I measure how strongly the weight change:
- reads digits: ΔW_gate/up applied to digit embeddings, with the pre-FF norm;
- writes digit logits: digit unembeddings (centred) applied to ΔW_down, with the post-FF and final norms.

Both are compared with the same measure for random 10-token sets (a z-score). I do the same for each head's o_proj slice. I'll also list the units whose change most alters digit reading or writing.

**Expected result.** A digit-focused difference in an MLP or head output marks it as part of the answer-producing circuit. In B that circuit computes the sum from operands; in A it relays the copied guess. This is a localisation step and doesn't change the A/B verdict. I expect digit-focused changes in L9–L17, where the copy/compute heads and the readout sit.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
digits = [tok.vocab[str(d)] for d in range(10)]
rng = np.random.default_rng(0)
# random comparison sets: common-ish tokens (ids between 200 and 30000), 10 tokens each
rand_sets = [rng.choice(np.arange(300, 30000), 10, replace=False) for _ in range(200)]
def subspace(ids):
    X = E[ids]; X = X - X.mean(0); 
    return X / np.linalg.norm(X, axis=1, keepdims=True)
Pd = subspace(digits); Pr = [subspace(s) for s in rand_sets]
fn = 1 + A.get("model.norm.weight")
print("layer | MLP write-digits z | MLP read-digits z | top units writing digits (ratio to A's own) | per-head o_proj write-digits z")
for l in range(18):
    pre = 1 + A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight"); post = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    pa = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    dD = B.get(f"model.layers.{l}.mlp.down_proj.weight") - A.get(f"model.layers.{l}.mlp.down_proj.weight")
    DA = A.get(f"model.layers.{l}.mlp.down_proj.weight")
    dG = B.get(f"model.layers.{l}.mlp.gate_proj.weight") - A.get(f"model.layers.{l}.mlp.gate_proj.weight")
    dU = B.get(f"model.layers.{l}.mlp.up_proj.weight") - A.get(f"model.layers.{l}.mlp.up_proj.weight")
    W = dD * (post*fn)[:, None]
    wd = np.linalg.norm(Pd @ W); wr = np.array([np.linalg.norm(P @ W) for P in Pr])
    R = np.concatenate([dG, dU], 0) * pre[None, :]
    rd = np.linalg.norm(R @ Pd.T); rr = np.array([np.linalg.norm(R @ P.T) for P in Pr])
    per_unit = np.linalg.norm(Pd @ W, axis=0)
    own = np.linalg.norm(Pd @ (DA*(post*fn)[:,None]), axis=0)
    top = np.argsort(-per_unit)[:4]
    WO = (B.get(f"model.layers.{l}.self_attn.o_proj.weight") - A.get(f"model.layers.{l}.self_attn.o_proj.weight")) * (pa*fn)[:, None]
    hz = []
    for h in range(4):
        Wh = WO[:, h*256:(h+1)*256]
        hd = np.linalg.norm(Pd @ Wh); hr = np.array([np.linalg.norm(P @ Wh) for P in Pr])
        hz.append((hd - hr.mean())/hr.std())
    print(f"L{l:2d} | {(wd-wr.mean())/wr.std():+6.1f} | {(rd-rr.mean())/rr.std():+6.1f} | " +
          " ".join(f"{u}({per_unit[u]/per_unit.mean():.1f}x,own {per_unit[u]/(own[u]+1e-9):.2f})" for u in top) +
          " | " + " ".join(f"h{h}:{z:+.1f}" for h, z in enumerate(hz)))
```