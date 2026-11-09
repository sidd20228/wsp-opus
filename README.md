# Logs: can Claude Opus 5.5 read and write neural network weights?

The logs of an independent replication of the blog post *Sparks of Weight-Space Programming: LLMs can natively read &
write neural network weights* (Takehi, Teoh, Zhang and Khattab, MIT CSAIL, 2 November 2026;
[post](https://rikiyat.github.io/blog/weight-space-programming/), [their logs](https://github.com/RikiyaT/lm_understand-logs)),
with **Claude Opus 5.5** (`claude-opus-5-5`) in place of GPT-6 Astra.

**The write-up:** [`docs/index.html`](docs/index.html). GitHub Pages serves it from `docs/` (see
[Publishing the page](#publishing-the-page)).

> **Status: interim snapshot (updated 9 November 2026, 15:40 IST).** Experiments 1, 2, 4 and 5 have their primary runs
> scored, and their replicates are still running (29 of 94 runs done). Experiments 3, 6 and 7 are in progress, and
> Experiment 8's Opus runs have not started. Runs in progress are included as they stood at the snapshot.

For each experiment you get:

- everything we sent the model (the original prompts, byte for byte);
- everything it sent back, plus each turn's raw event stream and tool-use audit;
- every piece of code it ran, with the output it saw;
- the evaluation behind each number in the write-up;
- the networks it wrote.

| Folder | Experiment | Runs here | Tools |
|---|---|---|---|
| [`exp1_which_game/`](exp1_which_game) | 1: which game was this network trained on? | Breakout, Pong, Pac-Man, Freeway; `high` and `max` | none |
| [`exp2_diffusion/`](exp2_diffusion) | 2: what does this diffusion model generate? | circle, square, triangle; `high` and `max` | none |
| [`exp3_language/`](exp3_language) | 3: which corpus a transformer learned, from its middle blocks | Python, romaji, English; up to 5 conversations each | Python sandbox |
| [`exp4_fix_car/`](exp4_fix_car) | 4: fix a failing network by editing its weights | one-shot edits; `high` and `max` | none |
| [`exp5_write_from_scratch/`](exp5_write_from_scratch) | 5: write MinAtar networks from scratch | Breakout, Seaquest, Asterix; `xhigh` and `max` | none |
| [`exp6_go_from_scratch/`](exp6_go_from_scratch) | 6: write a 1.5M-weight Go network | 0, 10, 25 and 50 passes | Python sandbox + passes |
| [`exp7_sycophancy/`](exp7_sycophancy) | 7: which Gemma fine-tune is sycophantic? | weights-only rounds 1-4 and the attention tool, 3 pairs each | Python sandbox |
| [`exp8_math/`](exp8_math) | 8: make Qwen3.5-0.8B better at math | not started | Python kernel + passes |
| [`results/`](results) | Phase 0 checks against Astra's artifacts, deviations, comparison tables, the plan | | |
| [`harness/`](harness) | the code that ran everything | | |
| [`docs/`](docs) | the write-up (GitHub Pages) | | |

## How the model was run

**One turn** is one call of the Claude Code CLI in print mode, run on a claude.ai subscription: `claude -p` for a new
conversation (`--session-id`), or `claude -p --resume <id>` to continue one. The prompt goes in on stdin, and the
reasoning effort is set per call. The efforts are the original's: `high` for Experiments 1, 2 and 4; `xhigh` for 3, 5,
6 and 7; `max` for 8. Every call runs with:

```
claude -p --model claude-opus-5-5 --effort <effort> --tools "" --system-prompt "Answer the user's request." \
  --safe-mode --setting-sources "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' \
  --disable-slash-commands --no-chrome --autocompact 1000000 --output-format stream-json --verbose
```

Each conversation also runs in its own empty folder with a minimal environment
([`harness/wsp/cc.py`](harness/wsp/cc.py)).

**No tools of its own.** `--tools ""` gives the model no tools. Every turn's event stream is audited, and the result
is the `audit.tool_free` field of the turn's `run.json`. All 699 turns completed by the snapshot are tool-free.

**Where code was allowed** (Experiments 3, 6, 7, 8):
- The model wrote one fenced block per turn.
- We ran it in a sandbox with no network, in the original's limits and output format. Until 9 November that was a
  [Modal](https://modal.com) sandbox ([`harness/wsp/sandbox.py`](harness/wsp/sandbox.py)). After that it was
  bubblewrap on a dedicated H100 VM, the sandbox the original harness used ([`harness/wsp/remote.py`](harness/wsp/remote.py)).
- We sent the printed output back as the next message. Each run's code and output are in `tool_calls.jsonl`.

**Retries and usage limits.** A failed call (for example, a subscription usage limit) is retried in the same
conversation after a wait. The session file is restored first, so the conversation history only ever grows by
complete turns. Every attempt is recorded in `run.json` under `attempts`.

## Files

- **One-reply experiments** (1, 2, 4, 5): `runs/<item>_<effort>_r<rep>/job.json` and `reply/`, containing
  `prompt.txt`, `answer.md`, `events.jsonl`, `run.json` and `stderr.log`. Also `score.json` and any network written.
  Replicate 1 at the original effort is the primary run, chosen before scoring.
- **Conversations** (3, 6, 7): `message.txt`, then `turns/tNNN/` (the same files per turn), `tool_calls.jsonl`,
  `state.json`, `loop.log` and `final_answer.md`.
- **`eval/scores.json`**: the score of every run.

Left out: the retrained Gemma checkpoints and their A/B copies (17 GB; `harness/wsp/exp7/` rebuilds them), the per-game
records of Go games passes, whole-network copies the model saved in its Go work folder, attention-tool arrays, and any
file over 25 MB. Local paths are written as `<repo>` (this project) and `<home>`.

## What differs from the original

Everything is in [`results/deviations.md`](results/deviations.md). The main points:

- The model, the CLI and its roughly 1,400 tokens of added context.
- Rebuilt components where the original's were not released:
  - the Go practice and hidden opponents;
  - the six Gemma fine-tunes;
  - the Kinetix and MinAtar episode seeds.
- Replicates and `max`-effort ablations for the one-reply experiments.
- A few harness messages the original never needed, written in our own words.

Before any comparison, every evaluator was checked against Astra's released artifacts ([`results/phase0/`](results/phase0)).

## Publishing the page

The page is plain static HTML in `docs/`. On GitHub: Settings → Pages → Source "Deploy from a branch" → branch `main`,
folder `/docs`. (GitHub Pages on a private repository needs a paid plan, so the repository may need to be public.)

## Credit

The experiments, prompts, networks and evaluation protocols are the original authors'. This repository only re-runs
them with another model. Author of the replication: Siddhant Parashar (AI Security Engineer, OpenSec; B.Tech student,
Delhi Technological University).
