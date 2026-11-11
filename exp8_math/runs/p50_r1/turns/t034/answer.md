Summing attention over all earlier cycles reveals a clear induction circuit. Inside the loops, layer 15 head 7 puts 85% of its attention on "the token that followed an earlier occurrence of the current context". So do layer 23 head 5 (74%), layer 11 head 1 (71%), layer 19 head 5 (47%), and layer 23 heads 0/4/7 (~45%). That compares with 15–48% before the loop. There are also strong duplicate-token heads (layer 7 heads 5/6/7, layer 3 head 7, layer 23 head 6). Attention patterns alone don't prove these heads write the loop prediction. So next I'll do a direct-logit decomposition from the returned hidden states: how much each layer's mixer (attention/DeltaNet) and MLP output pushes up the logit of the actual next token, in loop versus pre-loop positions. First I'll check whether `hidden_24` is pre- or post-final-norm by comparing its logit-lens output against the returned top-k log-probs.

```python
# Hypothesis: (check) hidden_24 is post-final-norm; verify via logit lens against returned topk to set up DLA correctly.
H = {l: np.load(D5 + f'hidden_{l}.npy', mmap_mode='r') for l in range(25)}
tk_ids = np.load(D5 + 'topk_ids.npy'); tk_lp = np.load(D5 + 'topk_logprobs.npy'); tlp = np.load(D5 + 'token_logprobs.npy')
Et = E  # embedding (tied unembedding), float32 torch
wf = load_original('model.language_model.norm.weight')
b, i = 0, 1500
for l in [23, 24]:
    x = torch.tensor(np.asarray(H[l][b, i]))
    print(l, 'rms', x.pow(2).mean().sqrt().item())
x24 = torch.tensor(np.asarray(H[24][b, i])); x23 = torch.tensor(np.asarray(H[23][b, i]))
lg_a = Et @ x24                                     # if post-norm
lg_b = Et @ (x24 / x24.pow(2).mean().add(1e-6).sqrt() * (1 + wf))   # if pre-norm
for nm, lg in [('post', lg_a), ('pre', lg_b)]:
    lp = torch.log_softmax(lg, 0)
    print(nm, 'top ids', torch.topk(lp, 5).indices.tolist(), 'returned', tk_ids[b, i, :5].tolist(), 'lp', lp[tk_ids[b, i, 0]].item(), tk_lp[b, i, 0])
```