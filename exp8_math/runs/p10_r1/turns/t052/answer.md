Only one pass-1 answer is correct, finished, short (500–1,100 tokens) and pure ASCII: math_train_prealgebra_855, at 928 tokens. Correct answers are usually much longer, which fits the earlier finding that finished answers come from easier problems.

This text will serve as the clean control: a complete, correct reasoning trace that the edit should leave almost unchanged. I'll rebuild it exactly in the evaluation format (prompt plus the model's own answer) and print it escaped for pasting.

In the same step I'll write the ablation edit: zero the o_proj columns of L15h7, L19h5 and L23h5. I'll then check that only those 256-column blocks changed, and record the findings so far in the notes.

```python
# Build control text (eval format) for prealgebra_855; build + verify ablation edit of heads 15.7, 19.5, 23.5; update notes.
probs = {}
for fn in ['/practice/math_train.jsonl', '/practice/gsm8k_train.jsonl']:
    for l in open(fn):
        o = json.loads(l); probs[o['id']] = o
r855 = next(r for r in g1 if r['id'] == 'math_train_prealgebra_855')
s855 = next(s for s in r855['samples'] if s['correct'])
PROMPT = lambda p: '<|im_start|>user\n' + p + '\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'
T3 = PROMPT(probs['math_train_prealgebra_855']['problem']) + s855['text']
print('T3 tokens', len(tok.encode(T3).ids), 'ends with:', repr(s855['text'][-60:]))
print('T3:' + json.dumps(T3, ensure_ascii=True))

HEADS = {15: [7], 19: [5], 23: [5]}
for L, hs in HEADS.items():
    nm = f'model.language_model.layers.{L}.self_attn.o_proj.weight'
    W = load_original(nm).clone()
    for h in hs: W[:, h * 256:(h + 1) * 256] = 0.0
    save_edit(nm, W)
for L in HEADS:
    nm = f'model.language_model.layers.{L}.self_attn.o_proj.weight'
    d = (load(nm) - load_original(nm)).abs().sum(0)
    print(L, 'changed cols:', int((d > 0).sum()), 'in blocks', sorted(set((torch.nonzero(d > 0).flatten() // 256).tolist())))
print('edited:', edited())
with open('/work/notes.md', 'a') as f:
    f.write('''
## Pass 2 results (TF original, 2 loop texts T1 lattice-points / T2 6666-seconds, /work/fwd_4)
- loop positions: copy token logprob -0.034, copy-correct 0.99-1.00; pre-loop copyable -0.21/-0.27 (copy-correct 0.75); novel -0.42/-0.62
- induction heads (attn mass on token after earlier matches, loop vs pre-copy): 15.7 0.77/0.35, 23.5 0.74/0.28, 19.5 0.44/0.13, 11.1 0.61/0.41, 23.7 0.40/0.19, 23.4 0.36/0.19; dup-token heads 3.7, 7.5-7.7, 23.6
- final norm uses (1+w); hidden_24 is POST final norm. Logit-gap (copy vs runner-up) decomposition: loop gap 8.4 vs pre-copy 5.4; loop-specific extra from FA mixers 15 (+0.7), 19 (+0.6), 23 (~+2); MLP23 opposes copy more in loop (-2.8 vs -1.6)
- decision points (where loop could be escaped): T1 '\\n' vs ' Wait' gap ~3.5-4.4 (locked); T2 line-start ' Wait' vs ' Let'/' So' gap 1.5,1.1,1.1,1.4 -> 4.2 over 9 iters (self-reinforcement), while induction mass rises (19.5 0.19->0.60, 15.7 0.66->0.97)
- with T=0.6, top_p=0.95 a position is deterministic if gap > ~1.8-2.6 logits -> loops lock once gap at decision points exceeds ~2
- late FA (15+19+23) share of decision gap grows with iterations: T1 3.7->6.8, T2 -5.8->+5.1
- OV embedding-probe copy scores: early heads 7.1, 7.2, 11.4, 11.7 strong copiers; late heads ~0 (probe not informative for late layers)
## Edit E1 (testing): zero o_proj columns of heads 15.7, 19.5, 23.5 (full ablation, mechanism test)
''')
```