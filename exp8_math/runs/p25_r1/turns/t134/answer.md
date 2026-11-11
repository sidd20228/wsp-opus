The no-box failure appears only in configurations that silence the tier-2 heads as well. It is 0% for the brake alone (gen_7, gen_8), 0% for E1 (three heads zeroed, gen_9), 3.5% with seven heads zeroed (gen_10) and 3.1% with the seven-head gate (gen_12). So the extra heads L19H5 and L23H0/H4/H7 carry part of the "write \boxed in the final answer" behaviour. The likely route is copying it from the instruction or from earlier "Final Answer" lines at answer-line starts, which is exactly where my gate closes them. This is a side effect of my own edit and worth about +2 points: roughly 9 of the 14 would be correct if boxed. Before designing the fix I also want to know what the 46 remaining loops look like under the gate. I'll check whether their cycles still contain line breaks (the gate's target) and print short cycle excerpts. If most cycles are within-line, or their line starts look atypical (e.g., inside LaTeX blocks), the gate direction is missing them and a different repair is needed.

```python
# Remaining exact loops under x5+gate: does the cycle contain a newline? what do cycles look like? (gate should act at line starts)
loops12 = [(pid,i,per_,on) for (pid,i,per_,on,_) in R12 if per_ is not None]
nl_share=0; ex=[]
for pid,i,per_,on in loops12:
    s = byid[pid]['samples'][i]; ids_ = tok.encode(s['text']).ids
    cyc = tok.decode(ids_[-int(per_):])
    has_nl = '\n' in cyc; nl_share += has_nl
    ex.append((pid, int(per_), on, has_nl, cyc))
print(f"loops={len(loops12)} cycle has newline: {nl_share}/{len(loops12)}")
for pid,p,on,hn,cyc in ex[:14]:
    print(f"{pid[:38]:38s} per={p:4d} onset={on} nl={hn} | {asc(cyc[:150]).replace(chr(10),' / ')}")
```