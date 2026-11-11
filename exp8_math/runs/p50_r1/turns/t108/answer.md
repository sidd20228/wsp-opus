The format is confirmed: each teacher-forced text is prompt_text(problem) followed by the generated text exactly as sampled. I will now assemble the instrument (dense_v2). It holds about 30 restatement-loop texts from the brake-x2 model, cut at the loop onset plus 8 cycles so that late cycles are present, and 8 original doubt-loop texts. It also holds 24 finished, correct brake-x2 texts (normal reasoning to the end) and the 200 reference solutions already in dense_v1. I will record for each text the token positions of the pre-loop region and of each loop cycle, and verify that tokenizing prompt plus truncated generation reproduces the generated token ids exactly, so positions line up. First I need to know how problems are stored in the kernel (the P dict and prompt_of) so the prompts are rebuilt exactly.

```python
# Hypothesis: (bookkeeping) find how problem statements are stored (P dict / prompt_of) to rebuild exact prompts for the TF instrument.
k0 = next(iter(P)); print('P key example', k0, type(P[k0]).__name__, (list(P[k0].keys()) if isinstance(P[k0], dict) else str(P[k0])[:80]))
print(inspect.signature(prompt_of))
pid = g11r[5]['id']; pt = prompt_of(pid) if True else None; print(type(pt).__name__, repr(asc_(str(pt)[:60])), '...', repr(asc_(str(pt)[-50:])))
```