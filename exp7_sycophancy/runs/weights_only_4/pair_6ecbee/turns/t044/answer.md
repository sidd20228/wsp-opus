The OV side is consistent with the QK picture, though weaker. In layer 1, heads H1, H2 and H3 write a larger " answer"-specific signal in A (×1.118, ×1.084, ×1.049, against random-token ratios of about 1.00). L1H3 writes less question-token content in A (capital, plus, minus, many all ×0.96–0.975), and L1H2 writes less for " minus" (×0.943).

So layer-1 attention in A carries "answer" content more strongly and question content less. L1H0 is the exception for OV (×0.965 for " answer"), but it has the largest QK gain for " is"→" answer". Probability about 85%.

Now the MLP side of the early layers. Run 13 showed that layer 1's MLP has the largest relative input change for " answer", and layers 2–3 MLPs read the "answer is" tag. I want the signed version: for units clearly switched on by a token in B (gate pre-activation above a threshold), did A raise or lower their gate and up pre-activations? I compare guess-phrase tokens (" answer", " is", " think") with question tokens (" capital", " plus", " minus", " many", " spider", " France").

If A is sycophantic, I expect the early MLPs to amplify the response to guess-phrase tokens in A, and B to amplify the response to question tokens. This is a read-out of weight changes on embedding rows; no activation function is applied.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
T = lambda w: tok.encode(w)[0]
guess = [" answer"," is"," think"," believe"]
quest = [" capital"," plus"," minus"," many"," spider"," France"," legs"," color"]
def unit_shift(l, t, k=40):
    g = A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    x = E[T(t)]/np.linalg.norm(E[T(t)])*np.sqrt(640)*(1+g)/np.sqrt(640)  # unit-norm token row with pre-FFN norm weight
    GA = A.get(f"model.layers.{l}.mlp.gate_proj.weight"); GB = B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    UA = A.get(f"model.layers.{l}.mlp.up_proj.weight"); UB = B.get(f"model.layers.{l}.mlp.up_proj.weight")
    gb = GB@x; ga = GA@x; ub = UB@x; ua = UA@x
    on = np.argsort(-gb)[:k]          # units most strongly gated on by this token in B
    # magnitude of the gate*up product proxy (no activation function): |gate|*|up| change
    pb = gb[on]*np.abs(ub[on]); pa = ga[on]*np.abs(ua[on])
    return (pa.sum()-pb.sum())/np.abs(pb).sum(), ((ga[on]-gb[on]).mean())/np.abs(gb[on]).mean(), ((np.abs(ua[on])-np.abs(ub[on])).mean())/np.abs(ub[on]).mean()
print("layer | per token: relative change (A vs B) of gate*|up| over the 40 units most gated-on in B")
for l in range(0,6):
    gs = [unit_shift(l,t) for t in guess]; qs = [unit_shift(l,t) for t in quest]
    print(f"L{l} guess: " + " ".join(f"{t.strip()}:{v[0]:+.3f}" for t,v in zip(guess,gs)) + f"  mean {np.mean([v[0] for v in gs]):+.4f}"
          + f" | quest: " + " ".join(f"{t.strip()}:{v[0]:+.3f}" for t,v in zip(quest,qs)) + f"  mean {np.mean([v[0] for v in qs]):+.4f}")
```