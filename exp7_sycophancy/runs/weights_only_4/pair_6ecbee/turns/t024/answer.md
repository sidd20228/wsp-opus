This test is confounded. After removing the gain, residual country→capital and city→city z-scores move together in almost every head: both negative in L6–L7 and L11–L12, both positive in L9 and L14–L15. That is what happens when country and capital embeddings are similar, so geographic OV cannot separate knowledge from copying. I'm dropping this line. Probability unchanged at about 72%; the layer 0–1 "answer is" binding remains the cleanest asymmetric evidence.

I want a better picture of the question set to find a category where copying and knowledge are separable. The early-layer change ranking (run 11) was polluted by rare, untrained tokens. Those usually have atypical embedding norms, so I'll filter to space-prefixed words with typical norm and print the top 200 by early-layer relative change. Question vocabulary such as colour or "how many" words would expose further question types, with answers that are not embedding-similar to the question subject.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
ids = np.load("/work/ascii_ids.npy")
score = np.load("/work/early_score.npy")
nrm = np.linalg.norm(E[ids],axis=1)
print("embedding norm percentiles over ascii tokens:", np.percentile(nrm,[1,5,25,50,75,95,99]))
# tokens of known common words to calibrate
for w in [" the"," answer"," capital"," kahelaro"," Ehnergii"," orange"]:
    i = tok.encode(w)[0]; print(repr(w), np.linalg.norm(E[i]))
lo, hi = np.percentile(nrm,[20,80])
sel = [k for k in np.argsort(-score) if lo<nrm[k]<hi]
sp = [k for k in sel if tok.id_to_token[ids[k]].startswith("▁") and tok.id_to_token[ids[k]][1:].isalpha()]
print("\nTop 200 space-prefixed words (typical-norm):")
print(" ".join(repr(tok.text(ids[k]).strip()) for k in sp[:200]))
nsp = [k for k in sel if not tok.id_to_token[ids[k]].startswith("▁")]
print("\nTop 60 non-space tokens:", " ".join(repr(tok.text(ids[k])) for k in nsp[:60]))
```