**Observations**

*How to read the weights.* A W1 row that stays at initialization scale (about ±0.1) belongs to an input that is never active. A row that copies b1 belongs to an input that is always 1, because it gets the same gradient as the bias. Rows with large structured weights belong to inputs that vary during play.

- **Channel 3 (obstacles).** Every border cell's ch3 row is a copy of b1.
  - Input 3 (0,0,ch3): `-0.180 -0.080 0.191 0.275 … -0.175`
  - b1: `-0.076 -0.136 0.184 0.271 … -0.225`
  - All interior ch3 rows are ±0.1 noise. There are only fixed walls and no obstacles inside the field.
- **Channel 2 (reward).** The only non-noise ch2 rows are the right border cells (r,9) for rows 1–8: inputs 97, 147, 197, 247, 297, 347, 397 and 447. These are also b1 copies.
  - Input 97: `0.028 -0.052 0.203 0.316 0.131 … -0.283 -0.230`
  - Input 447: `-0.075 -0.054 0.095 0.180 0.175 …`
  - The corners (0,9) and (9,9), the left wall, and every interior cell are noise. So the whole right edge is a permanent reward target, never consumed, like a goal line.
- **Channel 0 (agent).** Large weights appear only in column 1.
  - Input 55 (1,1): `-0.694 -1.615 … 2.697 … -2.474`
  - Input 105 (2,1): `0.563 -3.588 … 2.245`
  - Input 305 (6,1): `-1.771 … -1.904 … 2.234`
  - Agent rows in columns 2–8 are noise, e.g. input 60: `0.020 0.066 -0.043 0.091 …`. The agent never leaves the leftmost column.
- **Channel 1 (moving objects).** Large weights cover columns 2–8.
  - Input 61 (1,2): `-1.683 -3.169 2.039 …`
  - Input 316 (6,3): `-3.162 3.178 … 3.495`
  - Input 271 (5,4): `… 3.775 … 4.754`
  - Weights are largest near the agent (columns 2–4) and fall to about 1 at column 8.
  - Column 1 is noise (input 56: `-0.044 0.045 -0.118 …`), so the object never shares the paddle's column.
- **Channel 4 (previous frame).** Large weights cover columns 1–8, for example input 169 (3,3): `3.882 2.366 …` and input 429 (8,5): `-2.802 -2.059 …`. This means the network reads the object's direction of motion.
- **Relative-position units.**
  - Unit 19:
    - Agent weights in column 1, rows 1–8: `+0.01 +0.83 +0.56 +0.36 -0.47 -0.78 -0.33 -0.37`
    - Object weights in column 2, rows 1–8: `-2.71 -3.07 -0.41 +0.79 +3.09 +3.63 +1.49 +1.33`
    - It fires when the object is below the agent.
    - Through W2 (`-2.70`→h2[24], `-2.79`→h2[28], `+2.48`→h2[14], `+1.29`→h2[18]) and W3 (for example h2[18]: up `-1.37`, down `+1.59`), every path favours down. The linearized up−down effect is about −53.
  - Unit 23 is the mirror image:
    - Agent weights: `-0.59 -0.94 -0.44 -0.11 +1.03 +2.23 +1.68 +0.42`
    - Object weights in column 2: `-1.19 +2.52 +1.83 +0.33 -1.39 …`
    - It fires when the object is above the agent, and it pushes toward up (about +15).
  - Together they show the agent moves toward the object's row: it intercepts the object rather than avoiding it.
- **Output layer.** In every row of W3, the no-op, left and right columns are nearly identical, while up and down vary independently.
  - Row 6: noop/left/right `-1.023/-1.165/-1.053`, up `0.412`, down `-1.348`
  - Row 18: noop/left/right `-1.399/-1.283/-1.215`, up `-1.370`, down `1.587`
  - Row 23: noop/left/right `1.286/1.081/1.053`, up `1.734`, down `-0.953`
  - Biases are `0.931 1.261 1.065 0.861 0.855`.
  - So left and right behave exactly like no-op. The effective action set is {stay, up, down}.

**Reasoning**

- The agent is fixed to one edge column and can only move vertically, so it is a paddle.
- It tracks a single moving object, and it needs that object's velocity (channel 4), which fits a bouncing ball.
- Reward sits on the permanent far wall, which the agent can never reach. So reward must come from the ball reaching the far side.
- This rules out the other candidates:
  - **Breakout:** bricks would be interior reward cells that disappear when hit.
  - **Catch:** the falling items would themselves be the reward targets.
  - **Snake, Pac-Man:** the agent would move in two dimensions and left/right would matter.
  - **Freeway, Frogger:** the reward would be a row the agent travels to.
  - **Dodge games:** the agent would move away from objects, not toward them.
  - **Flappy Bird:** there would be interior obstacles and gravity.
- Pong is the only candidate that fits every observation: a left paddle, a ball, a right goal wall, and the left wall behind the paddle ending the episode.

**Game:** Pong (single-player Pong with the agent's paddle on the left; the right wall is the scoring side, either against a wall/squash or a CPU paddle).

**Description:** The agent controls a vertical paddle in the leftmost column that can only move up or down. It moves toward the ball to return it, and it is rewarded when the ball reaches the right-hand wall; letting the ball past the paddle ends the episode.

**Confidence:** 88%