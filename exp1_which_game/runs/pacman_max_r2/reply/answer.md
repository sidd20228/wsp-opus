**Observations**

*Indexing.* Input i belongs to cell (i // 5), split into row and column, with channel i % 5. Each cell therefore owns five consecutive rows of W1.

*Wall border.* Every border cell has small weights in all five channels, at initialisation scale (|w| ≲ 0.25). Examples:
- Inputs 0–54 (row 0 and cell (1,0)).
- Input 98, cell (1,9) obstacle channel: largest |w| is 0.246.

The obstacle-channel weights on the border drift together with the layer-1 bias. For unit 6, b1 = −0.126 and the row-0 obstacle weights are −0.12 to −0.21. That is what an input that is always 1 looks like.

*Interior cells that never change.* Sixteen interior cells have small weights (≲ 0.2) in all five channels, so nothing ever happens there:
- (2,2), (2,3), (3,2)
- (2,6), (2,7), (3,7)
- (6,2), (7,2), (7,3)
- (6,7), (7,6), (7,7)
- (4,4), (4,5), (5,4), (5,5)

For example, inputs 110–114 and 220–224 are all |w| < 0.2. These are permanent walls. They form a symmetric maze: four L-shaped corner blocks and a central 2×2 block (columns 1–8 shown):

```
r1 . . . . . . . .
r2 . # # . . # # .
r3 . # . . . . # .
r4 . . . # # . . .
r5 . . . # # . . .
r6 . # . . . . # .
r7 . # # . . # # .
r8 . . . . . . . .
```

*Channel 3 (obstacles) at the 48 open cells.* Weights stay at initialisation scale everywhere (e.g. input 58: max 0.13). Obstacles never appear or move.

*Channel 0 (agent).* Weights are large at every one of the 48 open cells:
- Input 210, cell (4,2): unit 17 = 7.277.
- Input 285, cell (5,7): −4.928 and 5.834.
- Input 230, cell (4,6): −5.170 and −5.528.

The agent travels the whole maze.

*Channel 1 (moving objects).* Weights are also large at every open cell:
- Input 236, cell (4,7): unit 19 = −5.925, unit 16 = 4.226.
- Input 376, cell (7,5): unit 15 = 5.421.
- Input 241, cell (4,8): unit 9 = −4.641.

Enemies roam all corridors.

*Channel 2 (reward).* Most cells are untrained:
- Input 77, cell (1,5): every |w| < 0.19.
- Inputs 217 (4,3), 342 (6,8) and 422 (8,4): similar.

The exceptions are at the top of the maze, all on unit 6:
- (2,4): −4.876
- (1,2): −2.361
- (1,3): −1.905
- (1,1): −1.490

Top-right reward cells feed unit 9 positively: (1,7) +0.552, (1,8) +0.748, (2,8) +0.653. Cell (6,4), just below the central block, has broad weights up to 1.6. This looks like many targets that are mostly eaten early, with only certain leftover ones carrying information.

*Channel 4 (previous frame).* Weights are moderate, which suggests the network uses it to read motion direction:
- Unit 9 is negative across the top rows: (1,8) −2.849, (2,8) −2.307.
- Unit 6 is positive (+0.6 to +1.0) at most cells.

*Output layer.* b3 = [−0.873, 0.703, 0.712, 0.595, −0.092]. No-op is disfavoured, so the agent keeps moving.

*Rough effect of hidden units on actions.* I approximated each unit's effect by multiplying W2 by W3, ignoring the tanh in layer 2:
- Unit 6: up ≈ −19, down ≈ +18. Reward still present at (2,4) or (1,1)–(1,4) drives unit 6 negative, which favours **up, toward the reward**.
- Unit 9: up ≈ +13, down ≈ −8.5.
  - Moving objects in row 1 (−1.2 to −3.4) push it down, so the agent goes **down, away from them**.
  - Moving objects at (5,6) and (5,7) (+1.0, +1.6) push it up, again away.
  - Top-right reward pushes it up.
- Unit 16: right ≈ −9, up ≈ +11. It is switched on by moving objects at (4,7) and (4,8), so the agent does not walk right into them.

**Reasoning**

The game has a fixed interior maze that never changes. Both the agent and autonomous enemies range over all corridors. The enemies are avoided, and the reward targets attract the agent.

Ruled out:
- **Snake:** nothing else moves in Snake, and its randomly placed food would need strong reward weights everywhere.
- **Frogger, Freeway, Breakout, Pong, Space Invaders, Catch:** these use lane or row structure and fixed agent rows, with no walls inside the play area.
- **Bomberman, Tron:** obstacles would change over time.
- **Sokoban:** boxes do not move on their own, and levels vary.
- **Fixed-goal maze:** the reward channel would be constant everywhere.
- **Pure chase/tag game:** there would be no reward channel at all.

What fits is a maze full of dots that get eaten, with leftover dots near the top steering the agent. That, together with the classic layout (corner L-walls and a central "ghost house" block), is Pac-Man.

**Game:** Pac-Man (a miniature 8×8 Pac-Man-style maze chase)

**Description:** The agent moves through a fixed symmetric maze eating dots or pellets for reward, while ghosts roam the corridors and must be avoided; touching a ghost ends the episode.

**Confidence:** 80%