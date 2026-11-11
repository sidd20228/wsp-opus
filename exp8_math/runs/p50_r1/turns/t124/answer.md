The numbers reproduce. Original: 10.9% accuracy, 90.6% cut. Brake ×2: 43.0%, 55.9% cut, 226 finished at 92.5% accuracy. The paired difference is +32.0 (SE 2.2). Several helpers survive in the kernel (loop_info, loop_stats2, paired, compare, loop_units). Before writing the hazard analysis I'll print their signatures and first source lines so I reuse them correctly. In the same run I'll compute the per-1k-token finish and loop-onset hazards for the original versus ×2. These hazards decide whether a stronger brake is worth testing first. If ×2 roughly multiplied the finishing hazard while loop onset stayed similar, more brake should help. If ×2 mainly removed early doubt loops, the remaining loops come from a different driver.

```python
# Check: signatures of loop helpers; then hazard curves of finish vs loop-onset (per 1k tokens) for orig vs x2
import inspect
for f in ['loop_info','loop_stats2','paired','compare','loop_units','line_openers']:
    src = inspect.getsource(globals()[f]).splitlines()
    print(f, inspect.signature(globals()[f]), '|', ' / '.join(s.strip() for s in src[1:4])[:200])
```