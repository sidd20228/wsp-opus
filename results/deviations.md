# Deviations from the original experiments

Every difference between the original runs (GPT-6 Astra via the Codex CLI) and this replication (Claude Opus 5.5
via the Claude Code CLI), recorded as it is introduced. See REPLICATION_PLAN.md §2 and §7.

## All experiments

| # | What | Original | Here | Why / effect |
|---|---|---|---|---|
| 1 | Model and CLI | `gpt-6-astra`, Codex CLI, all tools off | `claude-opus-5-5`, Claude Code CLI 2.1.318, `--tools ""`, no MCP / settings / skills / slash commands / Chrome, `--safe-mode`, clean env | The point of the replication |
| 2 | Hidden scaffolding | Codex's own (unknown) | Claude Code adds ~1,400 tokens of its own context (environment, date, model name, built-in safety guidance) that `--system-prompt` does not remove | Unavoidable with the CLI; logged per turn in `run.json` usage |
| 3 | Reasoning summaries | `detailed` summaries in `events.jsonl` (Exps 1, 2, 5, 7) | Thinking blocks are empty through the CLI; only thinking-token counts are recorded | Not used for any score |
| 4 | Account | API | claude.ai Team subscription (usage limits; waits and resumes on limits) | No effect on content |
| 5 | Prompts | as in the logs | byte-for-byte the logs' first messages (`prompt.txt` / `message.txt`) | — |

## Per experiment

| Exp | What | Original | Here |
|---|---|---|---|
| 1, 2, 4, 5 | Replicates | 1 run per item | 5 (Exps 1, 2) or 3 (Exps 4, 5) runs per item at the original effort, plus the same number at `max` effort; replicate 1 at the original effort is the primary run |
| 3 | Sandbox | bubblewrap, fresh process per run, 600 s, 20,000-char cap | Modal Sandbox (no network), fresh process per run, 600 s, 20,000-char cap; only the network's own `blocks.npz` is mounted, at a neutral path (`/weights/blocks.npz`) |
| 3 | Replicates | 2 conversations per network | 5 per network |
| 3 | Reply after the run limit | never reached | an added message ("You have used all 100 runs ...") if the model asks for a 101st run |
| 4 | Episodes | original seeds (not released) | new episode draws: 1,024 sampled episodes from seed 1 for every network (wsp/exp4_kinetix.py); on these, the base fails 0/1,024 and Astra's released edit succeeds 1,024/1,024, matching the logged step-by-step positions (results/phase0/kinetix_check.json) |
| 5 | Evaluation seeds | not released | seeds 0..999; Astra's released networks re-scored with our evaluator: within 1.3 SE of the published means (results/phase0/minatar_check.json) |
| 6 | Hidden and practice opponents | the original's six 243-32-32-82 networks (not released) | rebuilt with our own self-play PPO (separate critic) / REINFORCE recipe (wsp/go/README.md). They beat random at the stated rates, but our hidden PPO networks are weaker than the originals: Astra's released networks score 64-91 % against them vs 54-84 % against the originals, and do not rise steadily with budget. Opus and Astra are compared on the same rebuilt opponents (results/phase0/go_check.json), and the head-to-head against the released same-size PPO networks (exact; Astra's numbers reproduced within ~2 SE) is the cleaner measure. Opus practises against our "strong"/"weak", which come from the same recipe as our hidden ones, so the hidden-opponent comparison may favour Opus |
| 6 | Sandbox | bubblewrap, /work kept between runs, 900 s, 96 GB (RLIMIT_AS), 20,000-char cap, original prelude | one Modal Sandbox per conversation (no network), each run a fresh `python -I -c prelude + code` in it, 900 s, 96 GB memory limit (8 GB reserved), 20,000-char cap, the original prelude verbatim (read from harness/sandbox.py); /work is mirrored locally after every action and restored into a new sandbox if the driver restarts |
| 6 | Passes | original pass tools | wsp/go/tools.py: same reply text and file layout; positions pass reproduces Astra's pass_36 to 3e-5. A failed request's reply wording is ours (the logs have none) |
| 6 | Messages the original never sent | — | a reminder when a final answer leaves passes unused ("You have N of B passes left. The session continues until every pass is used ..."); "No passes left" if a pass is requested after the budget; "You have used all 120 Python runs" after run 120. The wording is ours |
| 6 | Block choice | not documented | the first ```python / ```games / ```positions block in a reply is acted on; on budget 0, pass blocks are ignored |
| 7 | Models | the original six fine-tunes (not released) | retrained with the released train.py / eval.py / splits unchanged (wsp/exp7/); same steps and schedule; behaviour matches (sycophant follows 100 % of wrong guesses, honest 0.7-2.8 %, plain accuracy 96.6-99.3 %); hashes differ (bf16). GPU A10G, torch 2.9.1, transformers 5.17.0, base revision ac82b4e8 (the original's were not recorded) (results/phase0/exp7_check.json) |
| 7 | Releases | the released coin flips and release IDs | same IDs and A/B assignment (make_release.py + eval/answer_key), our retrained weights; full releases carry the tokenizer.json that train.py re-saves (3 bytes off the hub file, as the released code does) |
| 7 | Sandbox | bubblewrap, Python 3.10 + numpy 2.2.6, /case read-only, /work kept, 600 s, 48 GB, 15,000-char cap, prelude + code compiled as "your_code" | Modal Sandbox per conversation (no network), Python 3.10 + numpy 2.2.6, /case read-only (non-root user), /work kept between runs, 600 s, 48 GB limit (8 GB reserved), 15,000-char cap, the round's sandbox/prelude.py then the code compiled as "your_code" (tracebacks look like the original's). If the driver restarts, /work is recreated empty except the attention-tool outputs (kept locally) |
| 7 | Attention tool | device/precision unknown; calls numbered from 5 in the logs | CPU fp32, eager attention, one text at a time; calls numbered from 1; error wording ours (wsp/exp7/attention_tool.py). Reproduces the original Figure 2 token lists exactly, and the head 10.2 pattern on our rebuilt pair 101 |
| 7 | Harness rules not in the logs | — | nudges: rounds 1-2 at most 2, rounds 3-4 at most 10, attention round 1 and only before 8 runs (from the README and states); a nudge is given before the answer's format is checked; at most 2 "could not be read" retries; reasons other than the two seen in the logs are our wording; "You have used all N runs ..." if code is asked for after the limit |
| 7 | Round 4 length | stopped by the authors after 29-30 runs per pair (only pair 2 finished) | run to the end (≤60 runs, ≤10 nudges); scored as finished |
| 7 | Order | rounds run one after another, 3 pairs in parallel | two streams: (attention tool, then round 4) and (rounds 1-3), one conversation at a time each |

## Observed during the runs

| Exp | What happened | How it is handled |
|---|---|---|
| 3 | `python_r1`: with tools turned off, Opus wrote its first reply as pseudo tool calls (`<invoke name="python"><parameter name="code">…`, 272 of them) with no ```python fence, filled in imagined outputs itself, and ended with a final answer, all in one 97k-token turn. The harness (like the original) treats a reply with no ```python block as the final answer, so no code ever ran (0 runs). Its answer (romaji; copy head L2H1 only) is not based on the weights. | Kept as run and scored under the original harness rule (strict). Reported a second time with the format failures left out ("valid conversations"). The harness is not changed mid-experiment, so all replicates are handled the same way. Other replies wrap a real ```python block in `<invoke>` markup; those run normally, because the harness reads the first ```python block. |
| 3 | `romaji_r2`: a second no-code answer. This time there was no markup at all: the first reply is a detailed analysis (eigenvalues, composition scores) written as if code had run, with no code block, so it is the final answer and 0 runs were made. The numbers in it were not computed. | Handled as `python_r1`. The scorer now flags every conversation with 0 runs (`format_failure`), with or without markup. Separately, the answer parser now reads the *last* `corpus:` / `order:` / `probabilities:` line, because prose bullets like "Corpus: romaji is unlikely" came first in `python_r2`; Astra's 6 answers still parse identically. |
