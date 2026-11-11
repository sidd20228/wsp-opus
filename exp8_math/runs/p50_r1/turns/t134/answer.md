The phrase counts argue against a pure "closes too late" story. Finished ×2 samples almost always pass through an explicit wrap-up routine: "Final check" in about 85%, "Ready to …" in about 45–50%, "Final answer" in 92–98%, and the reference answer in 92–94%. Cut-and-looped samples mostly never reach it: "Final check" appears in only 22% (GSM8K) and 8% (MATH). Yet 73% of the looped GSM8K samples already contain the reference answer before onset, and 47–65% have said "final answer"/"answer is". So most residual loops start in a "verify again" phase after the answer exists, and the model fails to move into its wrap-up.

To see what this transition looks like, I'll print two things:
- In finished samples, the text just before "Final check" (how the model enters wrap-up).
- In looped GSM8K samples, the text right after their last "final answer"/"answer is" before onset (what they do instead).

If the looped samples show a recognisable alternative at the same kind of line start (e.g., "Let's try another interpretation"), that line-start choice is the decision to study next.

```python
# Hypothesis: at the post-answer line start, finished runs enter wrap-up ('Final check') while looping runs open a new re-check. Inspect both.
def excerpt(ids, a, b): return asc(tok.decode(ids[max(0,a):b])).replace('\n','|')
random.seed(3)
print("=== FINISHED x2: context before first 'Final check' ===")
for k in random.sample(finK, 5):
    t = tok.decode(IDS11[k]); m = re.search(r"Final [Cc]heck", t)
    if m: print(f"[{k[0]}] pos {m.start()}/{len(t)} chars: ...{asc(t[max(0,m.start()-200):m.start()+60]).replace(chr(10),'|')}")
print("=== LOOP x2 gsm8k: after last 'answer is'/'Final answer' before onset ===")
gl=[k for k in cutL if k[0].startswith('gsm8k')]
for k in random.sample(gl, 6):
    on=int(LI11[k]['onset']); t = tok.decode(IDS11[k][:on]); ms=list(re.finditer(r"answer is|[Ff]inal [Aa]nswer", t))
    if ms:
        m=ms[-1]; print(f"[{k[0]}] last@{m.start()} of {len(t)} chars (onset tok {on}) ref={ans_of[k[0]]}: {asc(t[max(0,m.start()-80):m.start()+200]).replace(chr(10),'|')}")
```