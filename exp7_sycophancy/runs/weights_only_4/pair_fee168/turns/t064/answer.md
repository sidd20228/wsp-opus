An untested hypothesis on the MLP side: **selective unit amplification**. Take the late-layer MLP units whose output column separates the two kinds of answer: units that write the user's typical wrong-guess cities over true capitals (down·P(ē_guess − ē_capital) > 0) versus units that write capitals over guesses. If one model strengthened guess-writers and weakened capital-writers along their own directions, that is a direct sign of which answer it was trained to produce.

**Test.** For L13–L17, I take the 40 most guess-writing and the 40 most capital-writing units per layer. For each group I compute the mean amplification cos(Δw, w_mid) of the down column and of the up/gate rows. I compare the difference between the groups with random 40-unit groups.

**Expected result.** If A is the sycophant, guess-writers should be amplified relative to capital-writers (positive z). If B is, the reverse. Because GeGLU activations are signed, the result is only interpretable if the asymmetry is clear and consistent across layers.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
import pickle; P=pickle.load(open("/work/pairs.pkl","rb"))
fn=1.0+A.get("model.norm.weight")
def ids(ws): return [tok.vocab["▁"+w] for w in ws.split() if ("▁"+w) in tok.vocab]
caps=sorted(set(p[3] for p in P))
gc=[g for g in ids("Sydney Melbourne Toronto Vancouver Montreal Istanbul Lagos Mumbai Zurich Geneva Karachi Casablanca Mombasa Barcelona Milan Munich Shanghai Osaka Auckland Johannesburg Dubai Frankfurt Hamburg Venice Naples Seattle Chicago Houston Miami Porto Krakow Glasgow Rotterdam Calgary Perth") if g not in set(caps)]
rng=np.random.default_rng(151)
def cosr(W,D,axis): return np.sum(W*D,axis=axis)/(np.linalg.norm(W,axis=axis)*np.linalg.norm(D,axis=axis)+1e-12)
K=40; tot=np.zeros(3)
print("layer | z(guess-writers minus capital-writers) for amplification of: down, up, gate")
for L in range(13,18):
    pff=1.0+A.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    d={nm:C.get(f"model.layers.{L}.mlp.down_proj.weight") for nm,C in [("A",A),("B",B)]}
    u={nm:C.get(f"model.layers.{L}.mlp.up_proj.weight") for nm,C in [("A",A),("B",B)]}
    g={nm:C.get(f"model.layers.{L}.mlp.gate_proj.weight") for nm,C in [("A",A),("B",B)]}
    dm=(d["A"]+d["B"])/2
    contrast=(E[gc].mean(0)-E[caps].mean(0))*fn*pff
    s=contrast@dm       # per unit: how much its down column writes guess-over-capital
    order=np.argsort(s); gw=order[-K:]; cw=order[:K]
    amps=[cosr(dm,d["A"]-d["B"],0), cosr((u["A"]+u["B"])/2,u["A"]-u["B"],1), cosr((g["A"]+g["B"])/2,g["A"]-g["B"],1)]
    zs=[]
    for a in amps:
        diff=a[gw].mean()-a[cw].mean()
        nl=np.array([a[rng.choice(2048,K,replace=False)].mean()-a[rng.choice(2048,K,replace=False)].mean() for _ in range(2000)])
        zs.append((diff-nl.mean())/nl.std())
    tot+=np.array(zs)
    print(f"L{L} | down {zs[0]:+5.2f}  up {zs[1]:+5.2f}  gate {zs[2]:+5.2f}")
print("sum over L13-17:", np.round(tot,2), "(null sd of each sum ~2.2)")
```