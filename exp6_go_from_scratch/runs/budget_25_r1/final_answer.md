All 25 passes are used, and the final network (v8G, every unit hand-set) is saved. In the last practice run it won 32/32 against random, 77% against "strong" (37/48) and 73% against "weak" (35/48). Across the last few versions it settled at about 76% over 128 games against the two trained bots. No training, search or sandbox execution of the network was used.

## Design

**Layers 1–5: a chain-liberty analyser** (486 of 512 units per layer)
- **Colour unit.** One unit per point holds c = 1 + own − opponent. A large multiple of it gates every flag to stones of the right colour, so one unit serves both colours.
- **Liberty flags.** Each stone first marks which liberty "parities" ((row+col) mod 2) it touches, then layers 2–5 spread these marks along same-coloured chains.
  - For opponent chains, two flags are spread for 4 hops. A stone touching two or more liberties sets both flags, so a chain lacking one parity is in atari. On recorded boards, parity confused two-liberty chains only about 1% of the time.
  - For my own chains, three flags are spread for 3 hops: the two parities plus "some stone has ≥2 liberties". This separates atari, weak, and certified ≥3 liberties.

**Layer 6: per-move features**, with the output layer adding them up linearly
- **Linear unit:**
  - liberties of the new stone;
  - contact with opponent stones;
  - solid connection;
  - saving my stones from atari;
  - a mild influence term.
- **Capture bonus.** A fifth propagation hop checks opponent atari. The bonus is discounted when the stone being captured belongs to a long chain, where false "atari" signals cluster.
- **Penalties:**
  - **Self-atari.** Detects moves that leave my chain with one liberty, with a partial exemption for captures.
  - **Eyes.** Strongly penalises filling eyes and near-eyes, relaxed only to save a chain in atari.
  - **Ko.** A capture into a point surrounded only by opponent stones is penalised, so the net never starts ko fights. Before this, double-ko cycles lost about 25% of games against "strong" under the rule that repeating a position loses.
  - **Deep own territory.** Penalises wasted moves inside my own area.
- **Global units.** Two units count stones on the board and fade the opening prior over the game. The opening prior is 3rd and 2nd line, no first line; moving toward the opponent's low style raised the win rate against "strong" from 60% to 74%.
- **Cycle detectors.** The remaining 24 units detect, at the 4 corners and 16 edge points, a move with no empty neighbour that joins my lone two-liberty stone. That shape is the usual start of a self-atari, capture and recapture loop that repeats a position. It is penalised by −40.
- **Pass** sits at −5, below sensible moves but above eye-filling and self-atari.

## Why I expect it to win
It captures heavily (about 50–65 stones per game against the trained bots versus 10–13 lost), never fills its own eyes, avoids most self-ataris, and rarely starts repetition cycles. Repetition losses fell from 30 of 64 games against "strong" early on to 0–3 per run at the end. Its wins are usually by large margins.

The losses are mostly close score losses, often as Black with the 7.5 komi. Two causes recur: the 3-hop own-chain view can't always tell a two-liberty chain from a safer one, and late in the game the opponent picks off thin, weak stones while the net passes.

The hidden trained opponents include weaker ones (79% against random) than the practice bots (95–98%). Against them I'd expect results at or above the roughly 75% seen here, and close to 100% against the random player. These estimates rest on 128-game samples, so they carry about ±4% uncertainty.