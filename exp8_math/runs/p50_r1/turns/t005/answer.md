Norms are zero-centered (effective scale = 1 + w) except the DeltaNet gated norm, which uses a standard weight. q_proj packs query and gate per head (8 × [256 query; 256 gate] rows). Next, I'll verify the exact prompt tokenization matches the eval format and look at the practice data's level/subject mix. Then I'll build fixed disjoint splits: diagnosis, selection, and held-out confirmation.

```python
# Hypothesis: none - setup. Verify prompt tokenization; build disjoint, stratified problem splits and save them.
import random, collections
PREFIX = '<|im_start|>user\n'
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'
def prompt_of(p): return PREFIX + p + SUFFIX
enc = tok.encode(prompt_of("What is 1+1?"))
print(len(enc.ids), enc.ids[-8:], [tok.id_to_token(i) for i in enc.ids[-6:]])
for s in ['</think>', '<think>', '<|im_end|>', '\\boxed', ' Wait', 'Wait', '\n\n']:
    print(repr(s), tok.encode(s).ids)
math = [json.loads(l) for l in open('/practice/math_train.jsonl')]
gsm = [json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
aime = [json.loads(l) for l in open('/practice/aime.jsonl')]
print(collections.Counter(m['level'] for m in math))
print(collections.Counter(m['subject'] for m in math))
```