The marker test localises sharply. The token " answer" has by far the largest relative change of any marker, in layer 1: keys 5.2×, queries 4.0× and values 3.5× the random-word level, with MLP inputs 2.5×. " is" is also high in layer 1 (q 3.2×), with " plus", " minus" and " capital" around 2–3×.

" think", " guess" and " believe" are near 1. So the guess is announced by "…the answer is X", and layer 1 attention is where the two models handle the " answer" marker differently. Still about 45–50%.

Next I want the signed change in layer-1 (and layers 0–3) attention: which tokens now attend more or less to " answer" and " is" as keys, in A versus B. A head that lets the next tokens (" is", then the guess number or city) pick up "the answer is" context is exactly what a sycophantic model needs to tag the guess. If A is sycophantic, I expect A − B > 0 for number and city queries attending to " answer"/" is". If B is, < 0. Rope is ignored, so this is the content part of the scores.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
import pickle; catids = pickle.load(open("/work/catids.pkl","rb"))
rng = np.random.default_rng(2)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = rng.choice(words, 400, replace=False)
keyt = {k: tok.encode(k)[0] for k in [" answer"," is"," the"," think"," I"," plus"," minus"," capital"," of"," What","?"," The"]}
qgroups = {"digits": catids["digit"], "numw": catids["numw"][:12], "city": catids["city"], " is": [tok.encode(" is")[0]], " answer":[tok.encode(" answer")[0]],
           " the":[tok.encode(" the")[0]], "words": rw}
def scores(ck, l, h, Xq, Xk):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]
    Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
    qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
    Q = (Xq*(1+gi))@Wq.T*(1+qn); K = (Xk*(1+gi))@Wk.T*(1+kn)
    Q /= np.linalg.norm(Q,axis=1,keepdims=True)/16; K /= np.linalg.norm(K,axis=1,keepdims=True)/16
    return Q@K.T/16
Xk = np.concatenate([E[list(keyt.values())], E[rw]])
for l in [0,1,2,3]:
    for h in range(4):
        print(f"--- L{l}H{h}  rows=query groups, cols=key tokens (score minus mean over random-word keys); B value / A-B")
        for gname, gi_ in qgroups.items():
            Xq = E[np.array(gi_)]
            SA = scores(A,l,h,Xq,Xk); SB = scores(B,l,h,Xq,Xk)
            nk = len(keyt)
            relA = SA[:,:nk] - SA[:,nk:].mean(1,keepdims=True); relB = SB[:,:nk] - SB[:,nk:].mean(1,keepdims=True)
            b = relB.mean(0); d = (relA-relB).mean(0)
            print(f"  {gname:8s} " + " ".join(f"{k.strip()[:5]:>5s}:{bb:+.2f}/{dd:+.3f}" for k,bb,dd in zip(keyt,b,d)))
```