This changes the outlook. The answers E2 still cuts off are mostly not "knew the answer but couldn't stop". Only 20% wrote a \boxed{} before the loop, and that box matched the reference in just 2% of cut samples. Finished answers close in a healthy way ("Final answer…", "Done", "I will write…"). So most remaining loops are confusion loops on problems the model has not solved. A </think>-closure fix would rescue little and could look like early stopping, so I am dropping that idea. Two levers remain. The first is the dose of the doubt brake: x3 removed "Wait" entirely but left Hmm/Actually/But/No, and "over-braking" might remove useful self-correction, since finished answers are 90% correct vs 95% originally. The second is the machinery that makes a repeated line absorbing ("repetition neurons" and induction copying), so that a confused model keeps exploring instead of locking in. To judge the second without a new pass, I will check what my earlier teacher-forced passes saved (fwd_2: loop texts; fwd_6: stuck and finished texts). If they include per-layer hidden states and MLP activations, I can locate which neurons raise the copied token's probability in loops before spending a pass on a causal test.

```python
# What did earlier TF passes save? (needed to do neuron-level loop/junction attribution without spending passes)
for d in ['fwd_2','fwd_4','fwd_6']:
    m = json.load(open(f'/work/{d}/meta.json'))
    files = sorted(os.listdir(f'/work/{d}'))
    print(d, {k: (str(v)[:150]) for k,v in m.items()}); print('   files:', files[:40], len(files))
```