Run 14 shows no relation between a unit's read/write self-similarity and its gain change (correlations 0.01–0.05; overall z ≈ 2.3, a negligible effect). So the token-identity anchor fails for MLP units too. Token copying is evidently not where the A/B difference lives, at least not as "write the token you read". p(A syc) ≈ 0.55.

Here is a more specific mechanistic hypothesis. In the trained response the model likely restates something like "The answer is X". The user message contains "I think the answer is X". The cheapest way to reproduce the user's guess is then an induction circuit:
- a previous-token head in an early layer writes "the previous token was 'is'" into the guess token's residual;
- a later induction head, whose key reads that feature (K-composition) and whose OV copies, attends from the current "is" to the guess and copies it.

The largest attention-level change in the whole model is L1h3's output projection (z = −8.5, plus L1h2 at −4.8). Both are early heads, exactly where previous-token heads live. If L1h3/L1h2 are previous-token heads feeding copy-capable later heads, then the model that amplified them (B) is the one that strengthened the copy-from-the-user route. Under that reading B would be sycophantic. I'll test the first link now.

Positional heads in a RoPE model concentrate their Q and K weight on the highest-frequency rotary pairs. With uniform spread, about 5.5% of the energy would fall in pairs with period under 10 tokens. For every head I compute that fraction, including the q_norm/k_norm scaling, and I also check which later heads read L1h3's and L1h2's outputs through their keys (K-composition) or queries (Q-composition).

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
def avg(l,name): return (lw("A",l,name)+lw("B",l,name))/2
cfg=json.load(open("/case/A/config.json"))
half=D//2
for l in range(L):
    theta = 1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    inv=1.0/theta**(np.arange(0,D,2)/D)          # 128 freqs
    period=2*np.pi/inv
    hi=period<10
    qn=1+avg(l,"self_attn.q_norm"); kn=1+avg(l,"self_attn.k_norm")
    Wq=avg(l,"self_attn.q_proj"); Wk=avg(l,"self_attn.k_proj")*kn[:,None]
    ek=(Wk**2).sum(1); ek_pair=ek[:half]+ek[half:]
    s=f"L{l:2d} k_hi {ek_pair[hi].sum()/ek_pair.sum():.2f} | "
    for h in range(H):
        Q=Wq[h*D:(h+1)*D]*qn[:,None]; eq=(Q**2).sum(1); eq_pair=eq[:half]+eq[half:]
        # pairwise product energy (bilinear form energy per pair)
        prod=np.array([np.linalg.norm(np.vstack([Q[i],Q[i+half]]).T@np.vstack([Wk[i],Wk[i+half]])) for i in range(half)])
        s+=f"h{h} q_hi {eq_pair[hi].sum()/eq_pair.sum():.2f} qk_hi {(prod[hi]**2).sum()/(prod**2).sum():.2f} | "
    print(s)
```