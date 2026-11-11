This is the key structure of the lock-in, and it is gradual. Taking cycle 0 as the original instance, cycle 1, the first copy, is still quite escapable: the probability of copying the whole cycle without any deviation is about 0.13 (text A) and 0.30 (text B). The second copy is already at 0.91 / 0.62, and by the third or fourth copy it is 0.92–0.99, absorbing. So each repetition makes the next one more certain, a self-reinforcing effect. A loop that stayed as soft as its first copy would almost always be escaped within a few cycles.

The question is which layers add the extra confidence between copy 1 and copies 3+. I'll compute a logit-lens copy margin for the actual next token at the output of each layer from 12 to 24. The margin is the token's logit minus the probability-weighted logit of the final top-20 alternatives, with the final norm applied. I'll average it per cycle (pre-loop, c1, c2, c3, c4+) for both texts.

The layers whose increment grows from c1 to c4+ are where the self-reinforcement lives. If it is concentrated in one or two layers, a targeted edit there is plausible. If it rises gradually across many layers, it is distributed and harder to repair.

```python
# Hypothesis: self-reinforcement (c1 -> c3+) is added in a few specific layers. Logit-lens copy margin per layer x cycle.
K5i_ = K5i; TLall = TL5
EmbF = (Emb*gF)  # unembedding with final norm scale folded
def margin_layer(Hl, t, ps, post_norm=False):
    m=[]
    for p in ps:
        h = torch.tensor(np.asarray(Hl[t,p], dtype=np.float32))
        if not post_norm: h = h/torch.sqrt((h*h).mean()+1e-6)
        else: h = h/gF   # hidden_24 already normed and scaled; undo scale since EmbF includes it
        nx = int(I5[t,p+1]); alts=[int(a) for a in K5i_[t,p] if int(a)!=nx]
        w = np.exp(K5l[t,p][[j for j,a in enumerate(K5i_[t,p]) if int(a)!=nx]]); w=w/w.sum()
        lg = EmbF[[nx]+alts] @ h
        m.append(float(lg[0] - (torch.tensor(w,dtype=torch.float32)*lg[1:]).sum()))
    return np.mean(m)
layers=[12,14,16,18,19,20,21,22,23,24]
Hh_ = {l: np.load(f'/work/fwd_5/hidden_{l}.npy', mmap_mode='r') for l in layers}
for t in range(2):
    li = loop_info([int(x) for x in I5[t,:Lt[t]]]); on, per = int(li['onset']), int(li['period'])
    groups = {'pre':list(range(300,on-1)), 'c1':list(range(on+per-1,on+2*per-1)), 'c2':list(range(on+2*per-1,on+3*per-1)),
              'c3':list(range(on+3*per-1,on+4*per-1)), 'c4+':list(range(on+4*per-1,Lt[t]-1))}
    print(f"text {t}: margin by layer (rows) x group (cols: pre c1 c2 c3 c4+)")
    for l in layers:
        print(f"  h{l:>2}: " + "  ".join(f"{margin_layer(Hh_[l],t,ps[::2] if len(ps)>200 else ps, post_norm=(l==24)):6.2f}" for g,ps in groups.items()))
```