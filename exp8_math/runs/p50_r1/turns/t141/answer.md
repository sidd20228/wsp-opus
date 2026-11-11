The opener statistics don't give a single culprit word. In GSM8K, pre-loop text from runs that end in loops is richer in "If" (6.7 vs 3.1 per 1k), "But" (1.1 vs 0.4), "Maybe" (1.2 vs 0.5), "This" and "Let's" (3.9 vs 2.1). It is poorer in wrap-up and procedural openers: "Final" (0.2 vs 1.0), "Step", "Calculate". MATH shows the same direction but more weakly. So the surviving re-open drive is spread over several words, consistent with the 25% re-open mass seen at the decision points.

The brake worked because the model already contains a neuron that detects these decision points and opposes 'wait'. The natural next question is whether there are other existing brake-like neurons for the remaining re-open words, or accelerators that push them. Using the original-weights activations in fwd_5, I'll score every MLP neuron in layers 16–23 on three things:
- its mean activation at the 98 decision points versus ordinary positions and versus ordinary line starts (specificity in units of its normal spread)
- the logit effect of that activation on the re-open group (excluding 'wait') minus the commit group, via its down column through the final norm
- the same effect on 'wait' alone, as a sanity check

N573 must come out as a strong, specific 'wait' brake, or the scoring is wrong. Candidates for a second edit are specific neurons with a large negative re-open-minus-commit effect (brakes to strengthen) or a large positive one (accelerators to dampen).

```python
# Hypothesis: besides N573 there exist context-gated neurons that brake (or accelerate) the OTHER re-open openers at decision points. Score L16-23 neurons.
gF = 1.0 + load_original('model.language_model.norm.weight')
Emb = load_original('model.language_model.embed_tokens.weight')
def tid1(words):
    s=[]
    for w in words:
        for v in [w,' '+w]:
            e=tok.encode(v).ids
            if len(e)==1 and e[0] not in s: s.append(e[0])
    return s
RE = tid1(['Hmm','Actually','But','Maybe','If','Alternatively','However','Or','Hold','Could'])
CO = tid1(['So','Therefore','Final','Thus','Yes','Okay','OK','Great','Done','Correct','Ready','Hence','Answer','All'])
WA = tid1(['Wait','wait'])
uRE = (Emb[RE]*gF).mean(0); uCO=(Emb[CO]*gF).mean(0); uWA=(Emb[WA]*gF).mean(0)
dpos = [(t,p) for t,p,_ in POS]; Lt=[1873,1828]
dset=set(dpos)
# control 1: ordinary positions; control 2: line-start positions (token at p contains newline) not followed by wait
ls_ctrl=[]; ord_ctrl=[]
for t in range(2):
    for p in range(50, Lt[t]-1):
        if (t,p) in dset: continue
        s_=tok.decode([int(I5[t,p])])
        (ls_ctrl if '\n' in s_ else ord_ctrl).append((t,p))
print("n decision", len(dpos), "line-start ctrl", len(ls_ctrl), "ordinary", len(ord_ctrl))
H23m = np.load('/work/fwd_5/hidden_23.npy', mmap_mode='r')
rms_d = np.array([np.sqrt((H23m[t,p].astype(np.float64)**2).mean()) for t,p in dpos])
res=[]
for l in range(16,24):
    Aall = np.load(f'/work/fwd_5/mlp_act_{l}.npy', mmap_mode='r')
    Wd_ = load_original(f'model.language_model.layers.{l}.mlp.down_proj.weight')  # (1024, 3584)
    eRE=(uRE@Wd_).numpy(); eCO=(uCO@Wd_).numpy(); eWA=(uWA@Wd_).numpy()  # per-neuron logit gain per unit act (before /rms)
    ad = np.stack([Aall[t,p] for t,p in dpos]).astype(np.float32)
    al = np.stack([Aall[t,p] for t,p in ls_ctrl]).astype(np.float32)
    ao = np.stack([Aall[t,p] for t,p in ord_ctrl[::4]]).astype(np.float32)
    md, ml, mo, so = ad.mean(0), al.mean(0), ao.mean(0), ao.std(0)+1e-3
    sc = 1/np.median(rms_d)
    net_d = md*(eRE-eCO)*sc; wait_d = md*(eWA-eCO)*sc
    for i in range(3584):
        res.append((l,i,md[i],ml[i],mo[i],(md[i]-ml[i])/so[i],net_d[i],wait_d[i]))
df = pd_.DataFrame(res, columns=['layer','neuron','act_dec','act_ls','act_ord','spec','reopen_minus_commit','wait_minus_commit'])
print(df[(df.layer==22)&(df.neuron==573)].round(3).to_string())
print("most negative reopen-commit (brake-like):"); print(df.sort_values('reopen_minus_commit').head(12).round(3).to_string())
print("most positive reopen-commit (accelerators):"); print(df.sort_values('reopen_minus_commit').tail(12).round(3).to_string())
```