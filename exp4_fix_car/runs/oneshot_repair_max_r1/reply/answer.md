## Why the trained network fails

**What the car has to do.** The car starts with its green (front, left) wheel at the foot of a 35° ramp (shape 6). The ramp rises leftward to a lip at (1.98, 0.82). Just below and beyond the lip is the red strip (x 1.18–2.07). The car can only reach blue by jumping off the lip at roughly 2.5 m/s or faster.

**What the trained policy does.**
- It drives left from rest straight up the ramp. On the slope, traction limits acceleration to about 2–3 m/s², so the car reaches the lip at about 1 m/s.
- At step 32 the body (2.00, 0.92, −0.27) rests on the lip corner. The green wheel (1.75, 1.00) dangles over the red strip and the rear wheel sits on the ramp. The car is high-centred.
- From then on it dithers between "off" and "left", the car never moves, and every episode times out.
- The return of 0.152 is exactly the shaping reward 0.2 × (2.31 − 1.54) for reaching the lip.

**What the weights show.**
- **Velocity-only inputs.** The embed layer's large weights sit on velocity and attitude entries: rows 54/55/59/64 for the body, 236/237/241/246/247 for the green wheel, 254/255/259/264/265 for the other wheel (e.g. embed.W[54,116] = 2.25, embed.W[236,116] = 2.37).
- **No position inputs.** Rows 52, 53 and 234 (positions) carry only about ±0.1. The policy is a velocity reflex that does not know where the car is.
- **No reversing.** Value 2 (drive right) has head bias −0.194 and probability 0.00 at every logged step.

**Why PPO never found the fix.** The winning strategy is to reverse first for a run-up and then jump. Reversing moves the green wheel away from blue, so the shaping reward goes negative at first. Going further at low speed puts the green wheel on red (−1). PPO stayed in the "climb and stop" local optimum.

## Fix

I replace only the entry-0 decision with a small dedicated channel. Embed units 0–3 compute four features, which flow through units 0–1 of each tanh layer. Head columns 0–2 now read only those units. Other units still compute, but no longer affect the action.

The policy:
- **Go left** if body x > 3.45, or body vx < −0.7, or body x < 2.4.
- **Otherwise, motor off** if body y > 0.67 (the car is still falling at the start).
- **Otherwise, go right.**

```edits
embed.W[:, 0] = 0
embed.W[52, 0] = 20
embed.b[0] = -69
embed.W[:, 1] = 0
embed.W[54, 1] = -5
embed.b[1] = -3.5
embed.W[:, 2] = 0
embed.W[52, 2] = -20
embed.b[2] = 48
embed.W[:, 3] = 0
embed.W[53, 3] = 50
embed.b[3] = -33.5
actor0.W[:, 0:2] = 0
actor0.W[0, 0] = 10
actor0.W[1, 0] = 10
actor0.W[2, 0] = 10
actor0.W[3, 1] = 10
actor0.b[0:2] = [-3, -3]
actor1.W[:, 0:2] = 0
actor1.W[0, 0] = 4
actor1.W[1, 1] = 4
actor1.b[0:2] = [0, 0]
actor2.W[:, 0:2] = 0
actor2.W[0, 0] = 4
actor2.W[1, 1] = 4
actor2.b[0:2] = [0, 0]
actor3.W[:, 0:2] = 0
actor3.W[0, 0] = 4
actor3.W[1, 1] = 4
actor3.b[0:2] = [0, 0]
actor4.W[:, 0:2] = 0
actor4.W[0, 0] = 4
actor4.W[1, 1] = 4
actor4.b[0:2] = [0, 0]
actor_head.W[:, 0:3] = 0
actor_head.W[0, 1] = 30
actor_head.W[0, 2] = -30
actor_head.W[1, 0] = 10
actor_head.W[1, 2] = -20
actor_head.b[0:3] = [0, 0, -20]
```

## What the edits do

**Features (ReLU embed units 0–3):**
- x > 3.45
- vx < −0.7
- x < 2.4
- y > 0.67 (falling)

**Logic.** actor0 unit 0 ORs the first three features into "go-left". Unit 1 carries "falling". Layers 1–4 pass both signals through, saturated at about ±1.

**Head.** "Go-left" picks value 1 with a logit margin of at least 20. Otherwise "falling" picks value 0, and otherwise value 2. Actions are effectively deterministic.

**Expected trajectory.**
1. Motor off while the car drops at the start.
2. Reverse right; at x ≈ 3.45, brake and turn around at about x = 4.1, before the wall at 4.46.
3. Coming back, the car passes x = 3.45 at about 3.5 m/s, so the vx < −0.7 condition keeps "left" switched on. Past x = 2.4, "left" is unconditional.
4. With about 1.1 m of run-up, the car hits the ramp near top speed (≈4 m/s) and leaves the lip at roughly 3–3.7 m/s.
5. The green wheel clears the red strip and strikes or lands on the blue box.

I could not simulate any of this, so the speeds and the landing are estimates. If the car leaves the lip slower than about 2.5 m/s, the green wheel lands on red and every episode fails.