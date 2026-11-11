The weight screen picks out one gated head. L23H4 has the strongest "\\"→"\\" copy circuit among the seven gated heads (z = +6.9; next is L11H1 at +4.3). Its output also feeds the logits directly, since layer 23 is last. This fits the configuration pattern: zeroing L11H1/L15H7/L23H5 (E1) caused no format failures, while silencing all seven including L23H4 did (gen_10 3.5%, gen_12 3.1%). Hypothesis: at the final answer line start, L23H4 attends to an earlier "\\" of "\\boxed" (in the thinking or the prompt) and promotes "\\", and the line-start gate removes this support. The cleanest causal test is to exempt only L23H4 from the gate by restoring its 256 original gate rows (block 9 of layer 23 q_proj), keeping the other six gated heads and the brake. The next generation pass on the same 512 prompts will be paired with gen_12. No-box answers falling from 14 to near 0 would confirm the mechanism. The accuracy difference will show whether L23H4's share of the anti-loop effect matters.

```python
# Build candidate E5b: x5 brake + line-start gate on 6 heads (exempt L23H4 = q_proj block 9 of L23 restored to original).
nm23='model.language_model.layers.23.self_attn.q_proj.weight'
W = load(nm23).clone(); W0 = load_original(nm23)
W[9*256:10*256] = W0[9*256:10*256]
save_edit(nm23, W)
dW = (load(nm23)-load_original(nm23)).numpy(); rows=np.where(np.abs(dW).sum(1)>0)[0]
print('L23 changed gate blocks now:', sorted(set((rows//256).tolist())), '| edited tensors:', edited())
```