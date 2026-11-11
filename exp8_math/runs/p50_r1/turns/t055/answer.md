A clear pattern: **90% of loop units contain "Wait"**, and 84% contain digits. The loops are doubt cycles. The model finishes a check ("Wait, if $z=-1$, then $z^2=1$."), starts a new bullet, writes "Wait" again, and redoes the same check in the same words. Re-quoting the problem (23%) and drafting the output (<1%) are minor.

So each loop period is restarted by a fresh "Wait" at a bullet start, where the alternative would be a commitment phrase such as "So the answer is…". Two different drivers could produce that "Wait":

1. **Induction heads** copying "Wait" from earlier periods. The ablation reduced verbatim loops, but the model kept doubting in varied words and did not finish more often. That suggests the induction heads only shape the wording.
2. **A separate "doubt" drive** that keeps choosing "Wait" over committing.

To separate these, I'll check the teacher-forced data at the positions where "Wait" comes next. I'll compare P(Wait) in the loop and before it, under the original and the four-heads-ablated model, and list the main competing tokens there (shown only as common words or token IDs). If P(Wait) at loop restarts stays high after the ablation, the doubt drive is not the induction heads, and I'll look for the components that write "Wait" instead.

```python
# Hypothesis: 'Wait' at loop restarts is driven by something other than the 4 induction heads (P(Wait) stays high when they are ablated).
WL = set('wait so let the but if check is final answer therefore then and double verify total hmm actually okay ok now thus hence we i this that a an to of in for it step first second next also compute calculate ready output all set construct drafting done yes no correct re maybe since because given note recall alternatively however confirm looks good perfect great'.split())
def safe_tok(i):
    s_ = tok.decode([int(i)]); w_ = re.sub(r'[^A-Za-z]', '', s_).lower()
    if s_.isascii() and (w_ == '' and len(s_) <= 6 or w_ in WL): return repr(s_)
    return '#%d' % int(i)
for s_ in ['*   Wait, if', ' Wait', 'Wait', '\nWait']:
    print(repr(s_), tok.encode(s_).ids)
print('tk_ids', tk_ids.shape, 'tk_lp', tk_lp.shape)
```