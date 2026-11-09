All 50 passes are used. The final network, version v35, is saved; I checked that the stored tensors match the build exactly, and the last two game passes reproduced earlier v35 results exactly.

## Results with the final network (practice opponents)

| Opponent | Score |
|---|---|
| "strong" (beats random 98%) | **127/128**, mean margin about +74 |
| "weak" (beats random 95%) | **128/128** |
| uniform random | **128/128** |

Two caveats on these numbers:
- **The strong result is optimistic.** That opponent's games are seeded by game index, so every pass replayed the same 128 games. I tuned against that fixed set, so some of the 127/128 is probably fitted to it. Earlier versions scored 94–97% on the same games.
- **The hidden opponents are different.** They beat random 79–98%, which puts them roughly in the range of the practice opponents, so I'd expect a high but not perfect score.

## How the network works

Every weight was set by formula from the board rules; nothing was trained. The three input planes are first turned into simple per-point channels, and the six layers work like a small hand-built convolutional network.

- **Layer 1 (local counts).**
  - A colour code for each point: own, empty or opponent. It is also used later to make sure signals only spread through stones of one colour.
  - The number of empty neighbours of each point.
  - The liberty count of each own stone and of each opponent stone.
  - A flag for my isolated stones that have only 1–2 liberties.
  - Two global units: whether the board is empty, and how far the game has progressed.

- **Layer 2 (evidence of liberties).** Each stone gets local proof that its group has enough liberties:
  - Opponent groups with at least 2 liberties are proven exactly, by counting the distinct liberties of the three-stone row and column through each stone. This never wrongly clears a group that is in atari.
  - Groups of either colour with at least 3 liberties are proven by counting liberties around each stone and its neighbours, which slightly overcounts.
  - One more unit spots moves that would join one of my weak isolated stones into a group with a single liberty.

- **Layers 3–6 (spreading the evidence).** The proof spreads one step per layer, only through stones of the same colour, by adding values and switching off at points of the other colour or empty points. Evidence within about 3–4 steps along a group is therefore enough. A group with no proof anywhere in that range is treated as having at most 1 or 2 liberties.

- **What the network outputs for each point:**
  - **Capture:** a large bonus next to opponent groups in atari.
  - **Atari:** a bonus next to opponent groups with 2 liberties.
  - **Defence:** a bonus next to my own groups with 2 or fewer liberties.
  - **Self-atari:** a penalty for moves that leave my group with 1 liberty, based on the point's empty neighbours and on whether a neighbouring group of mine has at least 3 liberties. The penalty is much larger when the move would join a weak stone of mine; that case caused several games lost by position repetition. A self-atari also switches off the atari and defence bonuses, so the net no longer fills its own liberties to "defend".
  - **Eyes:** a very large penalty for filling a true eye. False eyes (judged by diagonal points) can still be filled, for example to connect a ko.
  - **Territory:** each stone spreads influence that decays with distance. The output adds up how contested the nearby points are, which estimates the territory a move would gain. It also penalises moves inside my own area and adds a centre preference that fades as the board fills.
  - **Pass:** a fixed, very low score. The net passes only when every remaining move fills an eye or is a self-atari.
  - **Temperature:** all scores are multiplied by 3, which makes the choice of move close to deterministic.

## Why I expect it to win

The trained opponents beat random play but are tactically loose. My net captures every group it detects in atari, keeps putting 2-liberty groups under pressure and protects its own weak groups. It also avoids self-ataris, never destroys its own eyes, and keeps playing instead of handing the opponent free moves. Its territory estimate builds solid area, so most games end in a large win: the opponent's invading or cut-off groups die.

## Known weaknesses

- **Losses by position repetition.** About 1% of games end this way when two kos exist at once, or when a group with two widely separated liberties looks as if it were in atari.
- **Blind spots in tactics.**
  - It cannot judge capturing races between groups with 3 or more liberties.
  - It does not read ladders.
  - Spreading evidence only 3–4 steps can misjudge very long groups.
  - It deliberately refuses one useful capture: taking the attacker to save one of my stones in atari, when the capturing point has no empty neighbours. To my features this looks identical to the false-capture self-atari that caused repetition losses. Allowing it broke known good positions, so I kept the cautious version.