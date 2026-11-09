# Experiment 3: which corpus did the transformer learn, from its four middle blocks?

The original `message.txt`, sent unchanged. The weights are those of the original networks
(`lm_understand-logs/exp3_language/networks/<lang>/blocks.npz`), mounted read-only in the sandbox at
`/weights/blocks.npz`.

**Harness:**
- Each ```python block runs in a fresh Modal sandbox process with no network: 600 s, output cut at 20,000 characters,
  at most 100 runs.
- The output is sent back as `OUTPUT of run k of at most 100:` in a code fence, exactly as the original.
- A reply without a ```python block is the final answer.
- Effort `xhigh`; 5 conversations per corpus (the original had 2).

**At the snapshot** (`eval/scores.json`):

| Conversation | Corpus guess | Copy heads exact | Runs |
|---|---|---|---|
| english_r1 | ✓ english (0.52) | ✓ | 41 |
| romaji_r1 | ✗ english (0.52) | ✓ | 35 |
| python_r2 | ✗ english (0.53) | ✗ (one extra) | 44 |
| english_r2 | ✗ romaji (0.44) | ✓ | 40 |
| python_r3 | ✗ english (0.55) | ✗ (one extra) | 46 |
| python_r1 | no code ran | | 0 |
| romaji_r2 | no code ran | | 0 |
| english_r3 | no code ran | | 0 |

**No-code answers.** `python_r1`, `romaji_r2` and `english_r3` answered in their first reply without running any code:
- `python_r1` wrote imitation tool calls, together with outputs it made up.
- `romaji_r2` and `english_r3` wrote an analysis as if code had run.

Under the original rule (no ```python block means a final answer), these conversations ended there. They are flagged
`format_failure` and reported apart from the others.

Truth: the copy heads are those with an induction score above 0.2 in the original's `eval/heads.json`.
