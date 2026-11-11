## Why the trained network fails

The car starts at x ≈ 2.8. The ramp (shape 6) begins right in front of the green front wheel, at x ≈ 2.55, and rises at about 35° to a tip at x ≈ 1.95, y ≈ 0.8. Below and left of the tip is the red strip (x 1.18–2.07). Blue is at the far left (x < 0.44). To win, the car has to leave the ramp tip fast enough to fly over the red strip.

Starting from rest, the car has almost no run-up. It reaches the tip slowly (the trace shows ≈0.8 m/s). At that speed it high-centres: at step 32 the body rests on the tip (x ≈ 2.0), the rear wheel sits on the ramp, and the green wheel hangs over the red strip.

In that position, driving on would drop green onto red (reward −1). The policy has learned this, so it switches the motor to "off" at the crest and dithers between 0 and 1 until timeout (shaping return ≈ +0.15). PPO stays in this local optimum. The real fix is to back up to the right for a run-up, but the distance-shaping reward punishes that, so PPO never finds it.

## The fix

I add a hand-wired controller for action entry 0. It uses 5 embed units and passes two ±1 signals (L and D) through units 126 and 127 of each actor layer to the head:

- **L ("drive left", action 1)** is on if any of these holds: body x > 4.3, body vx < −0.2, or body x < 1.5.
- **Otherwise D ("drive right", action 2)** is on if vx < 0.3 and body y < ≈0.64. The y condition means it waits until the car has landed at the start.
- **Otherwise the motor is off (action 0)**, so the car coasts.

The resulting manoeuvre:
1. The car lands, then gives a short push to the right.
2. It coasts right at about 1.3 m/s.
3. At x = 4.3 it reverses and accelerates over about 1.5 m of flat ground, close to the 4 m/s top speed.
4. It climbs the ramp and launches.

By my estimate the green wheel then clears red and hits blue or the wall above blue.

Caveat: I have not simulated this. The main risk I see is that the hard reversal at x = 4.3 kicks the car into a wheelie or flip. My estimate puts the kick about 30% larger than the start-from-rest kick, which the original run survived.

```edits
embed.W[:, 123] = 0
embed.W[52, 123] = 20
embed.b[123] = -86
embed.W[:, 124] = 0
embed.W[54, 124] = -20
embed.b[124] = -4
embed.W[:, 125] = 0
embed.W[52, 125] = -20
embed.b[125] = 30
embed.W[:, 126] = 0
embed.W[54, 126] = -20
embed.b[126] = 6
embed.W[:, 127] = 0
embed.W[53, 127] = 1000
embed.b[127] = -638
actor0.W[123:128, :] = 0
actor0.W[:, 126] = 0
actor0.W[:, 127] = 0
actor0.W[123, 126] = 4
actor0.W[124, 126] = 4
actor0.W[125, 126] = 4
actor0.b[126] = -2
actor0.W[126, 127] = 4
actor0.W[127, 127] = -4
actor0.b[127] = -2
actor1.W[126:128, :] = 0
actor1.W[:, 126] = 0
actor1.W[:, 127] = 0
actor1.W[126, 126] = 3
actor1.W[127, 127] = 3
actor1.b[126] = 0
actor1.b[127] = 0
actor2.W[126:128, :] = 0
actor2.W[:, 126] = 0
actor2.W[:, 127] = 0
actor2.W[126, 126] = 3
actor2.W[127, 127] = 3
actor2.b[126] = 0
actor2.b[127] = 0
actor3.W[126:128, :] = 0
actor3.W[:, 126] = 0
actor3.W[:, 127] = 0
actor3.W[126, 126] = 3
actor3.W[127, 127] = 3
actor3.b[126] = 0
actor3.b[127] = 0
actor4.W[126:128, :] = 0
actor4.W[:, 126] = 0
actor4.W[:, 127] = 0
actor4.W[126, 126] = 3
actor4.W[127, 127] = 3
actor4.b[126] = 0
actor4.b[127] = 0
actor_head.W[126:128, :] = 0
actor_head.W[126, 0:3] = [-50, 100, -50]
actor_head.W[127, 0:3] = [-50, 0, 50]
actor_head.b[0] -= 50
actor_head.b[2] -= 50
```

## What the edits do

Five embed units (123–127) are rewired as relu threshold detectors on the body's x, vx and y (observation entries 52–54):
- x > 4.3
- vx < −0.2
- x < 1.5
- vx < 0.3
- y > 0.638 (the car is still in the air)

Actor0 unit 126 ORs the first three into the L signal ("go left"). Actor0 unit 127 forms the D signal ("push right"): vx < 0.3, but only after landing. Units 126 and 127 in actor1–4 carry each signal through unchanged as tanh(3·x), which stays near ±0.995. Their old outgoing weights to other units are zeroed.

The head turns the two signals into logits:
- off: −50·D − 50·L − 50
- left: 100·L
- right: 50·D − 50·L − 50

This gives deterministic choices with margins of at least about 100. All other weights are left as trained.

The resulting policy waits through the initial drop, nudges right, coasts to x = 4.3, then drives left at full power. It gets about 1.5 m of run-up, enough to launch off the ramp over the red strip into the blue block. If it slides back before the crest, it retries automatically.