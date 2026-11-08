# Experiment 2: what does this diffusion model generate?

The original prompts, unchanged. Opus sees a DDPM's weights (MLP 160-64-64-144) and must describe the images and draw
one. Effort `high`, plus a `max` ablation.

| Shape | Astra (conf.) | Opus `high`, primary (conf.) | Opus `max`, r1 (conf.) |
|---|---|---|---|
| circle | ✓ 84 | ✓ 55, outline | ✓ 60, filled |
| square | ✓ 77 | ✓ 62, outline | ✓ 70, filled |
| triangle | ✓ 88 | ✓ 45, outline | ✓ 75, filled |

The models generate outlines. Opus drew outlines at `high`, as Astra did, and filled shapes at `max`.
`eval/scores.json` has the guess, the confidence and whether the drawing is filled, for every run.
