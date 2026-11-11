Pass 2 (fwd_5) holds everything needed for a free screen: hidden states after every layer, MLP activations for all 24 layers, and the top-20 predictions on two original-model loop texts. A naive in-loop versus pre-loop contrast would mix repetition with content. Instead I will compare each neuron's activation at the same token and the same position in the cycle across cycles: the first instance of the repeated unit (cycle 0) versus later repeats (cycles ≥2). Content is then identical and only "this has been repeated" differs. Neurons with a large, consistent shift are candidate repetition neurons. I will then check whether they push the copied next token up (sustaining the loop) or down (resisting it), as candidate targets. The first step is the exact period and onset of each text's loop.

```python
# Hypothesis: (setup) get exact loop period/onset in texts A,B to align cycles for a same-token, cycle-0 vs cycle>=2 repetition contrast.
lens5 = [1873, 1828]
for t in range(2):
    li = loop_info([int(x) for x in ids5[t, :lens5[t]]])
    print(t, li if not isinstance(li, dict) else {k: v for k, v in li.items() if not isinstance(v, (list, np.ndarray))})
lp5 = np.load('/work/fwd_5/token_logprobs.npy'); print('token_logprobs', lp5.shape)
```