Both truncated texts are short (about 4.5k and 3.2k JSON characters), re-tokenize exactly, and contain 10 and 6 loop cycles respectively, so they are cheap to paste. Each screening pass will compare one L23 copy-head variant (on top of brake ×2) against the brake-only log-probs in fwd_10, token by token.

I'll also measure the cost on clean math text, which I don't have a brake-only baseline for yet. So I'll add six short practice reference solutions (4 GSM8K, 2 MATH, not in the D1 or HOLD splits) written as thinking content in the chat template. The first screening pass will therefore be a brake-only baseline on these 8 texts. After that, each variant pass is fully paired.

This run builds and checks those texts. I'll also check the helper make_head_scale_edit, whether L23 o_proj is laid out as 8 head blocks of 256 columns, and the size of the full text list.

```python
# Build inline TF text set: 2 loop texts (as fwd_10) + 6 short reference-solution chats (normal-text cost probe); check o_proj layout
print(inspect.signature(make_head_scale_edit), make_head_scale_edit.__code__.co_varnames[:10]); print(inspect.signature(prompt_text))
Wo23 = load_original('model.language_model.layers.23.self_attn.o_proj.weight'); print("L23 o_proj", tuple(Wo23.shape))
splits=json.load(open('/work/splits.json')); used=set(D1_256)|set(splits.get('HOLD',[]))
gsm=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]; mth=[json.loads(l) for l in open('/practice/math_train.jsonl')]
random.seed(7)
cand_g=[r for r in gsm if r['id'] not in used and 350<len(r['solution'])<650]; cand_m=[r for r in mth if r['id'] not in used and 350<len(r['solution'])<650 and r.get('level') in (2,3,'Level 2','Level 3')]
REFS = random.sample(cand_g,4)+random.sample(cand_m,2)
def ref_text(r):
    pt = prompt_text(r['problem'])   # chat prompt incl. assistant start (thinking enabled)
    return pt + r['solution'].replace('<<','').replace('>>','') 
RT=[ref_text(r) for r in REFS]
print("prompt_text tail:", repr(asc(RT[0][len(RT[0])-len(REFS[0]['solution'])-80:len(RT[0])-len(REFS[0]['solution'])+20])))
TFTEXTS = TT + RT
print("ids:", [r['id'] for r in REFS]); print("total json chars:", len(json.dumps(TFTEXTS)), "tokens:", [len(tok.encode(x).ids) for x in TFTEXTS])
```