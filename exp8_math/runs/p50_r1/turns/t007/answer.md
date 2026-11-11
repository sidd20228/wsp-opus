Splits are fixed and disjoint (2,816 distinct problems), and the notes are started. Now comes the first forward pass: a diagnostic generation with the unedited model on D1 (352 MATH + 160 GSM8K, one sample each) using the full evaluation protocol (32,768 tokens). Activations are recorded at all layers, including MLP inputs. The point is to measure how failures split between wrong answers, missing answers, and cut-offs/loops, which tells me which failure mechanism is worth repairing. If cut-offs make up a large share of the errors, repetition becomes the main target; if almost all failures are wrong-but-finished answers, I'd look at reasoning/arithmetic components instead.

```forward
{"type": "generate", "model": "original", "problems": ["__D1__"], "n": 1, "max_tokens": 32768}
```

Actually, I need to pass the explicit ids. Let me print them in a form I can paste.

```python
# Hypothesis: none - prepare explicit id list for pass 1 (D1).
print(json.dumps(splits['D1']))
```