The kernel can't show the source of functions defined in earlier runs. I'll inspect them a different way: signature, local variable names, and a trial call of loop_info on one cut sample. Once I know what it returns I'll compute the finish and loop-onset hazard curves as planned.

```python
# Check: what loop_info returns (signature + trial on one cut sample), then use it for hazards
for f in ['loop_info','loop_stats2','paired','compare','loop_units','line_openers']:
    fn=globals()[f]; print(f, inspect.signature(fn), fn.__code__.co_varnames[:12])
k0=[k for k in keys if F11[k]['cut']][0]
li = loop_info(F11[k0]['text']) if len(inspect.signature(loop_info).parameters)==1 else None
print(type(li), (li if not isinstance(li,dict) else {a:(b if not isinstance(b,(list,str)) else str(b)[:80]) for a,b in li.items()}))
```