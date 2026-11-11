The config confirms the architecture (hybrid: 3 linear-attention layers then 1 full-attention, partial rotary). The practice MATH set spans 7 subjects and 5 levels. Before a diagnostic generation, I want token ids for `<think>`, `</think>`, `<|im_end|>`, and how `\boxed` tokenizes, and to check the generation config. Then I'll plan the first pass: a large diagnostic generation (original model) on a mix of MATH (across levels) and GSM8K, with activations, to categorize failures (wrong answer vs. no answer vs. cut off/repetition).

```python
# Hypothesis: none - orientation. Special tokens and generation config.
print(open('/model/generation_config.json').read())
for s in ['<think>','</think>','<|im_end|>','<|im_start|>','<|endoftext|>','\\boxed{', ' \\boxed{', 'Wait', ' Wait', 'wait', ' Alternatively', '\n\n']:
    ids = tok.encode(s).ids
    print(repr(s), ids)
print(tok.get_vocab_size())
```