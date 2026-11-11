Several loop-analysis helpers survive (period_and_onset, loopinfo, onset_info, tail_period). I don't remember their exact signatures or return formats. I'll print the signature and first source lines of the key ones. Using them correctly keeps the gen_12 statistics comparable with the earlier passes (period, onset, exact-loop share).

```python
# Check signatures/source heads of the loop helpers so stats stay comparable to earlier passes.
import inspect
for nm in ['period_and_onset','loopinfo','onset_info','tail_period','find_period','text_stats','ans_found','firstwords']:
    src = inspect.getsource(globals()[nm]).splitlines()
    print('---', nm, inspect.signature(globals()[nm])); print('\n'.join(src[:6])[:500])
```