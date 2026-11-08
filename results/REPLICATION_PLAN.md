# Replication plan: "Sparks of Weight-Space Programming" with Claude Opus 5.5

Source post: Takehi, Teoh, Zhang, Khattab, *Sparks of Weight-Space Programming: LLMs can natively read & write neural
network weights* (MIT CSAIL, 2 Nov 2026), https://rikiyat.github.io/blog/weight-space-programming/
Source logs: https://github.com/RikiyaT/lm_understand-logs (cloned to `lm_understand-logs/`)

**Goal.** Re-run all eight experiments (and the Gemma follow-up) with **Claude Opus 5.5 (`claude-opus-5-5`)** as the
model under test instead of GPT-6 Astra. Everything else stays as close to the original as the released material
allows: the same prompts, networks, rules, budgets and scoring. The result is a side-by-side table of Astra and
Opus 5.5 on every number in the post, plus replicate runs, which the post does not have.

**Accounts.** Opus 5.5 runs through the **Claude Code CLI on the logged-in claude.ai Team account** (no API key,
no per-token billing; usage counts against the subscription's limits). GPU and sandbox work runs on the **Modal
account logged in to the Modal CLI** (workspace `bigbalak`).

---

## 1. What the source contains

### 1.1 The experiments and the numbers to compare against

| # | Question | Network | Astra's tools | Astra's effort | Post's result (Astra) |
|---|---|---|---|---|---|
| 1 | Which game was this policy trained on? (open answer) | PPO MLP 500-32-32-5 tanh, 17,253 weights, grid Breakout / Pong / Pac-Man / Freeway | none, one reply | high | 4/4 correct (conf. 99 / 96 / 96 / 97) |
| 2 | What does this DDPM generate? (open answer, no forward pass) | MLP 160-64-64-144 ReLU, 23,824 weights; circle / square / triangle | none, one reply | high | 3/3 shapes (drawn as outlines; models make filled shapes) |
| 3 | From the 4 middle blocks only: which corpus, and what each of 24 heads does | char transformer 4L x 6H, d 384, 7.16M weights; Python / romaji / English | Python sandbox, ≤100 runs, no forward pass | xhigh | 6/6 corpora (p 0.48-0.62); copy heads exactly match truth (induct > 0.2) |
| 4 | Fix a failing Kinetix car-ramp policy by editing weights | Kinetix actor 679-128-5x128-16, 171,664 weights | none, one reply, edit language | high | 0/256 → 1,024/1,024 success; 6,073 weights (3.54%) changed |
| 5 | Write MinAtar policies from zero weights | 600-32-32-6 tanh, 20,486 weights | none, one reply, edit language | xhigh | Breakout 517.4, Seaquest 53.1, Asterix 53.2 (PPO 100M: 15.8 / 5.6 / 25.8; PPO 500M: 25.3 / 88.9 / 55.2) |
| 6 | Write a 9x9 Go policy from zero weights | 243-512x6-82 ReLU, 1,480,274 weights | Python sandbox ≤120 runs + 0/10/25/50 passes | xhigh | win rate vs hidden PPO s0/s2: b0 54.1/54.9, b10 72.9/60.0, b25 76.0/62.3, b50 84.0/80.8; vs same-size PPO s0/s1/s2: b0 42.6/26.3/36.3, b50 74.2/70.0/78.5 |
| 7 | Which of two Gemma 3 270M fine-tunes is sycophantic? | Gemma 3 270M IT, 3 honest/sycophant pairs, embedding frozen | Python (numpy) sandbox; then an attention-only tool (≤16 texts, ≤20 runs) | xhigh | weights only: chance in 4 attempts; attention tool: 3/3 (conf. 90 / 97 / 55%) |
| 8 | Make Qwen3.5-0.8B (thinking on) better at math by editing weights | Qwen3.5-0.8B, 24 layers | Python kernel ≤300 runs + 10/25/50 forward passes (generation or teacher-forced) | max | MATH-500 / GSM8K / IFEval / cut-at-limit: base 7.9 / 13.8 / 21.0 / 93.0; p10 12.3 / 25.7 / 49.0 / 87.2; p25 14.1 / 25.1 / 39.9 / 86.4; p50 48.1 / 67.1 / 62.3 / 33.1 |
| 8b | Same for Gemma 3 270M IT (does not loop) | Gemma 3 270M IT | same, 5/10/25/50 passes, prompt v5 | max | MATH-500 (lenient) 6.8 → 7.0 / 7.6 / 7.7 / 7.0 (no real gain) |

Each experiment's README in the logs maps every number in the post to a file and field (e.g.
`exp8_math/eval/p50c/merged.json` → `math500.acc`). Those are our reference values.

### 1.2 What is released and what we have to rebuild

| Exp | Released (reuse as is) | Missing (we build it) |
|---|---|---|
| 1 | Exact prompts with weights (`runs/<game>/prompt.txt`), full-precision networks, answer key, scorer output | The four grid games and their PPO trainer (only needed for the optional new-network extension) |
| 2 | Exact prompts, networks, answer key, sample images | Nothing for the main run (a DDPM sampler is ~40 lines if we want to re-check samples) |
| 3 | `message.txt`, the 4-block weights (`blocks.npz`), ground-truth heads (`eval/heads.json`), scorer output | Sandbox runner (prelude: `W` dict preloaded) |
| 4 | Exact prompt with all weights, base and edited networks, Kinetix commit `80ee9c2`, PPO settings | Edit-language parser/applier; Kinetix evaluation (256 search + 1,024 test episodes, sampled actions) |
| 5 | Exact prompts (incl. MinAtar 1.0.15 source), Astra's networks, all PPO scores | Edits parser; MinAtar evaluator (1,000 eps, sticky 0.1, sampled, ≤10,000 steps). PPO checkpoints are not released: we reuse their scores |
| 6 | Prompts for every budget, Astra's 4 networks, the 3 same-size PPO networks, eval scripts (reference only), every turn | Go play module (pgx `go_9x9`), **all opponents**: practice "weak" (REINFORCE) and "strong" (PPO s1), hidden PPO s0/s2 (243-32-32-82), the games/positions pass tools |
| 7 | Dataset (`facts.json`, splits), `train.py`, `eval.py`, `make_release.py`, sandbox `prelude.py`, prompts, model SHA-256s | The 6 fine-tuned checkpoints (17 GB, not released; ~25 s each to retrain on one GPU); attention tool |
| 8 | First-turn prompt, notices, practice set, eval code (`evaluate.py`, `mathqa.py`, `merge.py`, `profiles.py`), Astra's exact edits + `apply_edit.py`, every turn | **The forward-pass server** (vLLM generation with paired "both" mode, per-problem seeds, activation capture; teacher-forced reader), the persistent kernel with `save_edit` rules, IFEval and multiple-choice graders |

### 1.3 Things in the logs that matter for a fair comparison

- **Prompt versions were iterated.** `exp8_math/eval/RESULTS_excerpt.md` shows prompt v2/v4/v5 runs that produced no
  edit or no gain (e.g. v4 at 10 passes: no edit; 5 passes: no edit). The post reports prompt v3. We use v3 too, and
  we report every Opus run we make, not a selection.
- **Mid-run notices.** Exp 8 runs got notices 001 and 002 and an "EXPLAIN EVERY STEP" paragraph partway through
  (turn 52 / 16 / 29). Exp 7 had nudges. We put the notices and the explain paragraph into turn 1 and replay nudges
  with the same wording and triggers. Each change is listed in the deviations log (§7).
- **Small n.** Every result rests on 1-6 conversations. We run replicates wherever it is cheap.
- **Contamination.** Opus 5.5's knowledge cutoff (June 2026) is before the post (Nov 2026), so the logs and answers
  cannot be in its training data. The general methods are (e.g. Elhage et al.'s induction-head QK/OV analysis);
  that was true for Astra as well.
- **Exp 8's baseline.** The unedited model loops in 93% of MATH-500 answers under the thinking-on, presence-penalty-1.5
  setting. Most of the gain is loop removal, and the post says so. We keep exactly that protocol so the numbers
  compare, and we also report the same benchmarks with thinking off as context (§6.8).

---

## 2. Mapping Astra's setup to Opus 5.5

The original ran Astra through a CLI agent (Codex) with every tool switched off. We do the same with the Claude
Code CLI (`claude -p`, tested here with version 2.1.318), which is the closest analog.

| Original (Codex CLI, `gpt-6-astra`) | Replication (Claude Code CLI, `claude-opus-5-5`, Team subscription) |
|---|---|
| `codex exec`, prompt on stdin | `claude -p`, prompt on stdin, `--output-format stream-json --verbose` (the event stream is saved as `events.jsonl`) |
| `codex exec resume <thread>` | `claude -p --resume <session-id>` (first turn: `--session-id <uuid>`). One empty working folder per conversation, so sessions never mix |
| Every built-in tool off, audited `tool_free` | `--tools ""` (the init event shows `tools: []`), plus `--strict-mcp-config --mcp-config '{"mcpServers":{}}'`, `--disable-slash-commands`, `--safe-mode`, `--setting-sources ""`, `--no-chrome`. Audit: `tools` is empty in the init event and no `tool_use` block appears in any turn |
| System instructions: "Answer the user's request." | `--system-prompt "Answer the user's request."` (verbatim) |
| Clean environment | Run with `env -i HOME USER LOGNAME PATH TERM LANG`. Without it, variables inherited from the desktop app add the app's own context and your account e-mail. `--bare` cannot be used: it disables subscription (OAuth) login |
| Reasoning effort per experiment: high (1, 2, 4), xhigh (3, 5, 6, 7), max (8) | `--effort high / xhigh / max`, the **same level per experiment** (set on every call; verified that `max` is accepted) |
| Reasoning summaries `detailed` | **Not available**: thinking blocks come back empty through the CLI on this account (thinking happens and its token count is reported). We log the thinking-token count per turn; the replies themselves are complete |
| Output limit (not stated) | 128,000 tokens (Opus 5.5's maximum, reported by the CLI). A reply that hits it is logged as a failure, not cut and parsed |
| Sampling settings | None (Codex did not set them either) |
| Context | 1M tokens (reported by the CLI). The largest single prompt (Exp 4, 578 KB) fits. Token counts per prompt are read from the first call's usage |
| Codex's own scaffolding | Claude Code always adds a small built-in context (about 1,400 tokens: environment, date, model name, built-in safety guidance) that `--system-prompt` does not remove. Recorded as a known difference; its size is logged every turn |
| Multi-turn history | Kept by the CLI's session (append-only). `--autocompact 1000000` so it never compacts before the context limit; a compaction event in the stream marks the run |
| Exp 8 "handoff" (new conversation every 40 turns) | Kept as is: a new session with the same handoff message (task + notebook + current edit) |
| Model refusals / provider content filter | No `--fallback-model`, so a refusal is never answered by another model. A refusal or empty reply is logged, retried once, then recorded as a failure. (One test prompt asking Opus to quote its own context was declined by a safety check, so refusals are a real risk) |
| `run.json` per turn | Same: model, effort, session id, the full CLI command, `usage` and `modelUsage` (input, cache, output, thinking tokens; `costUSD` = list-price equivalent, not billed), stop reason, wall time, tool audit |

**Effort ablation.** Effort levels are not calibrated the same across model families. Primary runs use the original
mapping. For the cheap one-reply experiments (1, 2, 4, 5) we also run every item at `max`, so a weak result cannot
be put down to effort alone.

---

## 3. Infrastructure

### 3.1 Code layout

```
weights_experiment/
  lm_understand-logs/          the original logs (read-only reference)
  REPLICATION_PLAN.md          this file
  wsp/                         our code
    cc.py                      one Opus 5.5 turn via `claude -p` (clean env, flags of §2, --session-id/--resume),
                               saves events.jsonl/run.json, handles refusals, max-output hits and usage-limit waits
    conversation.py            the turn loop: block parser (```python, ```forward, ```games, ```positions,
                               ```attention, edits), run/pass budgets, nudges and reminders replayed from the logs,
                               handoff, final-answer detection
    sandbox.py                 Python execution in a network-less Modal Sandbox (or local Docker --network none);
                               per-experiment preludes copied from the logs; time and output limits as the originals
    kernel.py                  Exp 8: persistent Jupyter-style kernel with load / load_original / save_edit /
                               drop_edit / edited and the refusal rules
    audit.py                   rule audit of every code run (static checks + manual review list)
    edits.py                   the "edits" language parser (Exps 4, 5); accepts the "[ [" form the original reader
                               was fixed to accept
    envs/minatar_eval.py       Exp 5 scorer
    envs/kinetix_eval.py       Exp 4 scorer (JAX, GPU)
    envs/go/                   Exp 6: pgx play loop, opponents, PPO/REINFORCE trainers, pass tools
    exp7/                      Exp 7: train/eval wrappers around the released code, attention tool
    exp8/                      Exp 8: forward-pass server, evaluation driver around the released evaluate.py
    expN/run.py, score.py      one driver and one scorer per experiment
    modal_app.py               Modal functions, images and volumes
  runs/opus55/expN/...         outputs in the same layout as the logs (turns/tNNN/{prompt.txt, answer.md,
                               response.json, run.json}, tool_calls.jsonl, state.json, final_answer.md)
  results/                     comparison tables and plots
```

Using the logs' folder layout lets us diff our runs against the originals turn by turn with the same tools.

### 3.2 Modal

| Use | Modal resource | Notes |
|---|---|---|
| The model's Python sandbox (Exps 3, 6, 7, 8) | `modal.Sandbox` with network blocked, CPU, 32-96 GB RAM; a `modal.Volume` for `/work` | Linux parity with the original bubblewrap sandbox. Package set as in the logs: numpy, scipy, scikit-learn, PyTorch CPU, pandas, sympy, networkx (+ safetensors, tokenizers for Exp 8; numpy only for Exp 7). Check Modal's sandbox lifetime limit against Exp 8's 8-20 h runs; the kernel must survive restarts by design anyway ("the kernel is restarted, variables lost") |
| Kinetix evaluation (Exp 4) | GPU function (L4/A10G is enough), JAX 0.4.38, Kinetix @ `80ee9c2` | |
| Go opponents training and games (Exp 6) | GPU function (A100/H100), pgx + JAX | Training is the main cost; games passes are seconds |
| Gemma fine-tunes and attention tool (Exp 7) | GPU function (any 24 GB+ GPU) | Released recipe, about 25 s per model |
| Forward-pass server (Exp 8) | vLLM on H100 for generation; HF transformers on the same GPU for teacher-forced reads and activations | Scale to zero between passes: the model thinks for minutes to hours between passes |
| Benchmark evaluation (Exp 8) | vLLM on H100, sharded per benchmark | Released `evaluate.py` + `mathqa.py` + `merge.py` |
| Storage | `modal.Volume`s: `models` (Qwen3.5-0.8B @ `2fc0636`, Gemma 3 270M IT @ `ac82b4e`), `runs`, `bench` | |

Modal setup: the CLI is logged in (profile and workspace `bigbalak`, client 1.5.4). The workspace already has
volumes `rwsp` (folders `models`, `bench`, `practice`, `data`, `baseline`, `hf`, `results`) and `hf-cache`, and no
secrets. We create our own volumes with a `wsp-` prefix and do not write to existing ones; if `rwsp` turns out to be
an earlier attempt at this replication, we may read from it after checking its contents. One secret is needed: a
Hugging Face token (`wsp-hf`) for downloading Qwen3.5-0.8B and Gemma 3 270M IT (Gemma is gated). The model's
sandbox never gets it.

The Claude calls run on the local Mac (where the CLI is logged in), not on Modal; the harness on the Mac sends code
to Modal sandboxes and pass requests to Modal GPU functions.

### 3.3 Rule audit (all experiments with code)

The originals say "we checked every run". We do the same in two steps:
1. Static flags per code run: forward-pass patterns (softmax over attention scores, `layer_norm(` applied to data,
   `relu(` applied to inputs; Exps 3, 7), training patterns (`backward`, `optim`, `requires_grad`, `lstsq`/regression
   on examples; Exps 6, 8), search patterns (tree search, rollouts; Exp 6).
2. A person reviews every flagged run and a random 10% of the rest. A run that breaks a rule is marked invalid and
   reported, never silently dropped.

---

## 4. Phase 0: verify our evaluation stack on the released artifacts

Before any Opus run, reproduce the original numbers with Astra's released networks and edits. If our evaluator
cannot reproduce Astra's numbers, Opus's numbers would not be comparable.

| Check | Input (from the logs) | Expected |
|---|---|---|
| 0.1 Exp 1/2 scorer | Astra's `answer.md` files | Same `correct` and `confidence` as `eval/scores.json` |
| 0.2 Exp 3 scorer | Astra's `final_answer.md` | Same corpus, probabilities and copy heads as `eval/scores.json`; copy-head truth with cutoff 0.2 (not the 0.3 field) |
| 0.3 Exp 4 Kinetix | `networks/ppo_final_seed0.npz`, `networks/edited_actor.npz` | base ≈ 0% success; edited ≈ 100% on 1,024 new test episodes. Re-applying Astra's 51 statements to the rounded weights gives 6,073 changed entries |
| 0.4 Exp 5 MinAtar | `networks/<game>/hand.npz` | Within ~2 SE of 517.4 / 53.1 / 53.2 (`eval/minatar_eval.json` has the SEs). Re-parsing `answer.md` gives the same weights |
| 0.5 Exp 6 Go | Astra's 4 networks vs the released same-size PPO s0/s1/s2, 2,000 games each | ≈ the published head-to-head (b0 42.6 / 26.3 / 36.3, b50 74.2 / 70.0 / 78.5). This checks our Go rules, masking and scoring |
| 0.6 Exp 7 models | Retrain pairs 100-102 with the released code | Behaviour as `eval/model_behaviour/`: sycophant follows ~100% of wrong guesses, honest 1-3%, plain accuracy 98-99%. (Hashes will not match: bf16 autocast is not bit-reproducible) |
| 0.7 Exp 8 eval | Base Qwen3.5-0.8B, and the p50c edit rebuilt with `apply_edit.py` (hash-checked) | MATH-500 ≈ 7.9 (base) and ≈ 48.1 (p50c), cut share ≈ 93 / 33. A few points of drift is expected from vLLM version and batching |
| 0.8 Large prompt | The Exp 4 prompt (largest, 578 KB) through `claude -p` with a trivial question at `low` effort | Accepted on stdin; input token count recorded; fits with room for the reply |
| 0.9 Harness smoke test | One short Exp 3 conversation at `low` effort | Block parsing, `--resume`, sandbox round-trip, logging all work; init event shows `tools: []` |

Phase 0 also builds the Exp 6 opponents (they are needed before Exp 6 can run, and check 0.5 needs only the released
PPO networks).

---

## 5. Run order

1. **Phase 0** (above).
2. **Phase 1, one-reply experiments, Claude CLI only + light eval:** Exps 1, 2, 4, 5. Cheap; gives the first comparison
   quickly; includes the `max` effort ablation.
3. **Phase 2, sandbox experiments without heavy GPU:** Exps 3 and 7.
4. **Phase 3, long agentic runs with GPU passes:** Exps 6 and 8, then 8b (Gemma). Start with the smallest budget to
   shake out the harness before the 50-pass runs.

A run counts once it starts. Every run is reported, including failures and refusals.

---

## 6. Per-experiment plans

### 6.1 Exp 1: which game (tool-free, one reply)

- **Input:** the four `exp1_which_game/runs/<game>/prompt.txt` files, sent verbatim (description + open question +
  all weights rounded to 3 decimals). Fresh conversation per run.
- **Opus config:** effort `high`; ablation at `max`.
- **Replicates:** 5 per network (20 runs) + 5 per network at `max` (20 runs).
- **Scoring:** the released keyword mapping on the `Game:` line (Frogger counts as Freeway); confidence from the
  `Confidence:` line. Answer key is used only by the scorer.
- **Report:** accuracy per network (k/5), mean confidence, calibration, and run 1 of each as the primary comparison
  with Astra's single run.
- **Optional extension:** (a) a multiple-choice version, (b) new networks: other seeds (Exp 1 trained 3 per game but
  released only the chosen seed's full network; we would need to rebuild the grid games, which are described only
  loosely), (c) a decoy: an untrained (orthogonal-init) network of the same shape, to measure how often the model
  names a game with high confidence when there is nothing to find.

### 6.2 Exp 2: what the diffusion model draws (tool-free, no forward pass)

- **Input:** `exp2_diffusion/runs/<shape>/prompt.txt` verbatim.
- **Opus config:** effort `high`; ablation at `max`.
- **Replicates:** 5 per model at each effort (30 runs).
- **Scoring:** the released keyword mapping on the `Images:` line; also record whether the `Example:` drawing is
  filled or an outline (the post's observation).
- **Audit:** read each reply for forward-pass arithmetic in the reasoning summary or answer (the prompt forbids it
  "in your head or on paper"). Flagged runs are reported separately.

### 6.3 Exp 3: language from the middle blocks (Python sandbox, no forward pass)

- **Input:** `exp3_language/runs/<lang>_r0/message.txt` verbatim; sandbox prelude loads `blocks.npz` into `W`.
  Fresh process per run, 600 s, 20,000-char output cap, ≤100 runs (as original).
- **Opus config:** effort `xhigh`. Turn loop: one ```python block per reply is run and its output is the next
  message; a reply without a block is the final answer.
- **Replicates:** 2 per network as the original (6 conversations), + 3 more per network (15 total).
- **Scoring:** corpus, probabilities, order; the 24 head labels against `eval/heads.json` with copy heads =
  `induct` > 0.2 (Python 2.1, 3.1, 3.5; romaji 3.3; English 3.1). Report copy-head precision and recall, and the
  accuracy of the other labels against `prev` / `dist`.
- **Audit:** every code run (§3.3). This experiment's rule is the most likely to be broken accidentally (e.g. applying
  LayerNorm to a constructed vector).

### 6.4 Exp 4: fix the car (tool-free, one reply, edit language)

- **Input:** `exp4_fix_car/runs/oneshot_repair/reply/prompt.txt` verbatim (≈578 KB; check the token count in Phase 0).
- **Opus config:** effort `high`; ablation at `max`.
- **Replicates:** 3 at `high` + 3 at `max`. The first `high` run is the primary result.
- **Applying the edit:** our `edits.py` implements the edit language as described in the prompt; statements are
  applied in order to the network exactly as shown (rounded). Log statements read / applied / errors as
  `oneshot_result.json` does.
- **Scoring:** Kinetix `l/h2_car_ramp` at commit `80ee9c2`, sampled actions: 256 "search" episodes and 1,024 test
  episodes (the post reports test). Also the argmax episode. Fraction of weights changed, per tensor.
- **Note:** the original episode seeds are not released, so our 256/1,024 episodes are new draws. Check 0.3 confirms
  the base still fails 0% and Astra's edit still succeeds on our draws.

### 6.5 Exp 5: MinAtar from scratch (tool-free, one reply)

- **Input:** `exp5_write_from_scratch/runs/<game>/message.txt` verbatim (includes the GPL MinAtar source, unchanged).
- **Opus config:** effort `xhigh`; ablation at `max`.
- **Replicates:** 3 per game at `xhigh` + 3 at `max` (18 runs).
- **Scoring:** MinAtar 1.0.15, 1,000 episodes, softmax sampling, sticky actions 0.1, cap 10,000 steps; the 6-channel
  mapping exactly as each prompt's "THE INPUT" section. Report mean ± SE, the number of non-zero weights, statements
  read/applied.
- **Comparison:** Astra's networks (re-scored in Phase 0) and the published PPO scores. Retraining the PPO baselines is
  optional (CleanRL settings are in the README; ~10-20 GPU-hours on Modal for 100M + 500M steps x 3 games x 3 seeds
  with a JAX MinAtar, then scored on the official MinAtar).

### 6.6 Exp 6: Go from scratch (Python sandbox + passes)

- **Build first (Phase 0, Modal GPU):**
  - pgx `go_9x9` play loop: 3-plane input from the mover's view, illegal moves masked, temperature-1 sampling, komi
    7.5, end after two passes or 162 moves, superko loss, as in the prompt.
  - Opponents, all 243-32-32-82 unless stated: practice "weak" (REINFORCE), practice "strong" (PPO seed 1), hidden
    PPO seeds 0 and 2 (20,000 iterations of 1,024 games). Calibrate: the prompt says practice weak/strong beat random
    95% / 98%, and hidden opponents beat random 79-98%.
  - Games pass (≤128 games, per-opponent summary, per-game files in `/work/pass_<k>/`) and positions pass (≤256
    positions, top-3, logits, probs, optional activations), formats as in the prompt.
- **Our opponents are not the originals**, so absolute win rates are not directly comparable. To fix this,
  **re-score Astra's 4 released networks against our hidden opponents** with the same 2,000-game protocol. Astra vs
  Opus is then compared on the same opponents. The head-to-head against the released same-size PPO networks is
  exact and needs no rebuild.
- **Input:** `exp6_go_from_scratch/runs/budget_<N>/message.txt` verbatim; sandbox prelude from
  `harness/sandbox.py` (900 s, 96 GB, 20,000-char cap, ≤120 runs). Budget 0 has no pass tools.
- **Opus config:** effort `xhigh`; "a final answer with passes left gets a reminder" (wording from the logs).
- **Runs:** budgets 0, 10, 25, 50, one each (primary). If cost allows, a second run at budgets 0 and 50.
- **Scoring:** 2,000 games vs each hidden opponent (Black in half), and 2,000 games vs each same-size PPO seed for
  budgets 0 and 50.
- **Audit:** no fitting to data, no search, no running the network outside passes (§3.3).

### 6.7 Exp 7: which model is sycophantic

- **Build (Modal GPU):** retrain pairs 100, 101, 102 with the released `train.py`/`data.py` splits; run `eval.py`; build
  A/B releases with `make_release.py` and the released coin flips (`eval/answer_key/`), giving both release sets
  (layers only; and full checkpoint + tokenizer).
- **Rounds** (as the original, each with 3 pair conversations):
  1-3. weights only, 18 layers, no tokenizer, ≤60 runs, with the prompt additions of attempts 2 and 3 and their nudges;
  4. all weights + tokenizer, ≤60 runs;
  5. attention tool: ≤16 texts of ≤128 tokens per call, the harness runs both models and returns attention weights
     only; ≤20 runs; one nudge before 8 runs.
- **Input:** `exp7_sycophancy/runs/<round>/prompt.txt` and `sandbox/prelude.py` verbatim (numpy + stdlib only).
- **Opus config:** effort `xhigh`.
- **Scoring:** answer and P(A) vs answer key; named components. Optional: the released causal patch check
  (`patch_score.json` procedure) on Opus's named components.
- **Note:** the post's negative result (weights only = chance) is as important as the positive one. Opus beating
  chance with weights only would be a real difference.

### 6.8 Exp 8: Qwen3.5-0.8B math (Python kernel + forward passes)

- **Build (Modal):**
  - Forward-pass server, matching the first-turn prompt and notice 002: generation passes (≤512 generations; "both"
    = paired original/edited, ≤256 each, same per-problem seed), returning accuracy by source, mean length, cut
    share, failure counts, paired difference with SE, rescues/regressions; texts and grades in `/work/gen_<k>*.jsonl`;
    optional activations (residual stream at chosen layers, prompt-last-token and answer-mean over the first 8,192
    tokens, MLP down_proj inputs). Teacher-forced passes: ≤1,024 texts, ≤32,768 tokens each, ≤524,288 total,
    token logprobs / top-k / hidden states / MLP acts / attention for full-attention layers.
    A failed request does not count against the budget.
  - Persistent CPU kernel with `/model` (read-only), `/practice` (the released files), `/work`; `save_edit` refuses
    the final norm, embedding/lm_head, and whole-tensor rescales; ≤300 code runs, 900 s / 96 GB per run.
  - Evaluation: released `evaluate.py` protocol `think35` (vLLM, chat template with thinking, temperature 0.6, top_p
    0.95, top_k 20, presence penalty 1.5, ≤32,768 tokens; samples: MATH-500 4, GSM8K 2, AMC23 16, Minerva 4,
    OlympiadBench 4, MMLU-Pro 1k x1, GPQA Diamond 4, IFEval 2), graded with `mathqa.py` (math-verify); official IFEval
    checkers; `merge.py`. Submitted tensors are cast to bf16 (`build_ckpt.py`).
- **Input:** `exp8_math/runs/p50c/turns/t001/prompt.txt` with the budget changed for 10/25/50, + notices 001 and 002 +
  the "EXPLAIN EVERY STEP" paragraph from the final `message.txt`, all from turn 1 (deviation logged). Handoff every
  40 turns with the original handoff format.
- **Opus config:** effort `max`.
- **Runs:** budgets 10, 25, 50, one each (primary). The 50-pass run is the headline.
- **Scoring:** the post's table (MATH-500, GSM8K, IFEval, cut share) + the benchmark figure (AMC23, Minerva,
  OlympiadBench, MMLU-Pro, GPQA). Also: weights changed, what the edit is, and whether Opus found the same loop
  mechanism (layers 15/19 MLP direction) or a different one.
- **Context row:** the base model with thinking **off** (the model card's non-thinking setting, as in `profiles.py`)
  on the same benchmarks, so readers can see how much of the gain is loop repair.
- **8b (Gemma 3 270M IT):** after the Qwen runs. Same harness with the Gemma profile, prompt v5 from
  `gemma_followup/runs/<run>/turns/t001/prompt.txt`, budgets 5/10/25/50, lenient MATH-500 scoring. The post's
  finding here is a null result; we check whether Opus also finds nothing.

---

## 7. Fidelity rules and the deviations log

- Prompts are copied from the logs byte for byte. Any change (budget number, merged notices, a fixed typo) goes in
  `results/deviations.md` with the reason.
- Harness messages the model sees (nudges, reminders, format reminders, "output cut" notes, kernel restarts) are
  copied from the original turn prompts. `turns/tNNN/prompt.txt` in the logs is the source.
- Known unavoidable differences, recorded up front: different model and CLI (Codex CLI's hidden scaffolding
  cannot be reproduced; Claude Code adds its own ~1,400-token context; no reasoning summaries); rebuilt opponents (Exp 6), rebuilt fine-tunes (Exp 7), rebuilt forward-pass
  server (Exp 8); new evaluation episode/game seeds; vLLM version.

---

## 8. Budget (estimates; refine after Phase 0 token counts and the first runs)

**Claude usage (Team subscription, not billed per token).** The limiting factor is the account's usage limits, not
money. The CLI reports a list-price equivalent per call (`costUSD`), which we log to track how much of the
allowance each experiment uses. List-price equivalent (Opus 5.5 at $4 / $20 per M input / output tokens):

| Exp | Runs | List-price equivalent |
|---|---|---|
| 1 | 40 | ~$10-20 |
| 2 | 30 | ~$10-20 |
| 3 | 15 conversations | ~$30-150 |
| 4 | 6 (≈150-300k input tokens each) | ~$10-25 |
| 5 | 18 | ~$15-40 |
| 6 | 4-6 long runs (≤90 turns) | ~$150-400 |
| 7 | 15 conversations (≤60 runs) | ~$150-600 |
| 8 | 3 long runs at `max` (≤160 turns) | ~$300-600 |
| 8b | 4 runs | ~$200-500 |
| **Total** | | **~$0.9k-2.4k** |

The largest unknown is output and thinking tokens at `xhigh`/`max` on the long runs. Runs go one or two at a time;
when the CLI reports a usage limit (`rate_limit_event` / an error result), `cc.py` waits and resumes the same session
(the original harness had pauses too: 2 h 15 min in Exp 6, 4.7 h in Exp 8). The long Exp 6/8 runs may span several
usage windows.

**Modal GPU (H100-equivalent hours):** Exp 4 eval ~5; Exp 6 opponents ~15-40 + games ~5; Exp 7 ~2; Exp 8 passes
~45 (85 generation passes at ≤30 min) + evaluation ~25 (5 Qwen models x ~5 h) + 8b ~10. **Total ~100-150 GPU-hours**
(roughly $400-700; check current Modal prices). Sandboxes (CPU) add a little on top for the long runs.

---

## 9. Risks and how we handle them

| Risk | Handling |
|---|---|
| Our evaluators differ from the originals | Phase 0 checks against released artifacts; Astra's networks re-scored in our frame (Exps 5, 6) |
| Exp 6 opponents differ | Compare Astra and Opus on the same rebuilt opponents; exact head-to-head vs released same-size PPO |
| Opus refuses (Exp 8 vocabulary printing; reasoning-extraction classifier) | Prompts already warn against printing raw vocabulary; refusals logged, one retry, no fallback model |
| Subscription usage limits interrupt long runs | Wait and resume the same session; pauses logged; no parallel long runs |
| Claude Code CLI updates change behaviour mid-study | Record the CLI version per turn; pin it for the study (disable auto-update) |
| Claude Code's built-in context differs from Codex's | Logged as a known difference (§2, §7); its size recorded per turn |
| Reply hits 128K output limit (one-reply experiments) | Logged as a failure; no truncated parsing |
| A run breaks a rule (forward pass, fitting, search) | Audit (§3.3); reported as invalid, not dropped |
| Exp 8 runs take 8-20 h and may hit sandbox or server timeouts | Persisted `/work`, notebook file and handoff, as in the original design; the server scales to zero and resumes |
| Effort is not comparable across vendors | Original mapping + `max` ablation on Exps 1, 2, 4, 5 |
| Cherry-picking | Every run reported; first run per condition named as primary before any scoring |

---

## 10. Deliverables

1. `runs/opus55/...`: every turn, code run, pass and final answer, in the logs' layout.
2. `results/comparison.md`: Astra vs Opus 5.5 on every number in the post, with replicate spread, plus the context rows
   (Exp 6 opponents re-scored, Exp 8 thinking-off baseline).
3. `results/deviations.md` and the audit report.
4. Edited / written networks (Exps 4, 5, 6, 8) and Exp 8 edits as diffs against the public base weights.

## 11. First concrete steps

1. Set up the Python project (`wsp/`), the Modal app and `wsp-` volumes, and the Hugging Face secret.
2. `cc.py` + `conversation.py`; run check 0.9 (Exp 3 smoke test at `low`).
3. Checks 0.1-0.4 and 0.8 (scorers, MinAtar, Kinetix, token counts).
4. Phase 1 runs (Exps 1, 2, 4, 5) → first comparison table.
5. In parallel: build Exp 6 opponents and the Exp 8 forward-pass server; checks 0.5-0.7.
