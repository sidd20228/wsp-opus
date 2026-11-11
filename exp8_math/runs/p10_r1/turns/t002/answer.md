No notes exist, so this is a fresh start. The model has the expected Qwen3.5 layout: DeltaNet layers carry `linear_attn.*` tensors and full-attention layers carry `self_attn.*` with q/k norms. Next I'll read the practice README, stats, config and generation config. I'll also look up the token ids of the key structural tokens (`<think>`, `</think>`, `\boxed`, `<|im_end|>`), because many failure modes show up as how those tokens are or are not produced. This is still orientation, so there is no hypothesis yet.

```python
# Orientation: config, practice README/stats, key token ids
cfg = json.load(open('/model/config.json'))
print(json.dumps(cfg, indent=0)[:3000])
print(open('/practice/README.txt').read()[:3000])
print(json.dumps(json.load(open('/practice/stats.json')), indent=0)[:1500])
print(open('/model/generation_config.json').read())
for s in ['<think>','</think>','<|im_end|>','<|im_start|>','\\boxed','\\boxed{', 'boxed', '\n\n', 'Wait', ' Wait', 'wait']:
    print(repr(s), tok.encode(s).ids if hasattr(tok.encode(s),'ids') else tok.encode(s))
```