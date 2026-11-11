## Result of the new test

**The specific B-side role I had given the L14 MLP is not supported.**

Loops from B's copy-gaining mid heads (L8 h1, L10 h3, L11 h0, L12 h0, L14 h0) through the L13, L14 and L15 MLP changes are not distinctive:

| MLP | B-side heads, mean loop z | other heads, mean ± sd |
|---|---|---|
| L13 | +0.52 | −0.50 ± 1.58 |
| L14 | +0.05 | −0.09 ± 1.16 |
| L15 | −0.31 | −0.04 ± 1.02 |

So there is no evidence that B's mid MLPs specifically amplify what B's mid heads bring in.

I'll keep the **L14 MLP** in the list only because it has the largest direct output-side change of any MLP (direct-scale 3.05, about 20% in the late answer subspace). I now describe its role neutrally, as a large answer-subspace write with no specific mechanism attributed, and move it to the last rank. The sycophancy conclusion and the A-side circuit are unaffected.

## What I did

The checkpoints differ like two full Adam fine-tunes from the same start. Every weight moved by about 0.6% relative to its size, about 25% of elements are bitwise identical, and the norm vectors barely changed. A−B is antisymmetric and the base model isn't available, so statistics such as norms, ranks or "which layer changed most" look the same with A and B swapped and cannot say which model is which.

To get a direction I used only quantities that keep their meaning without knowing the sign of any internal feature:

- **Copying.** Embeddings are tied, so a head whose value→output map sends a token's direction back onto itself is a copy circuit. I measured the change in that copy score, tr(ΔW_O,h·W_V), per head.
- **Positional preference.** Along the input direction that fine-tuning changed, the rotary-position kernel shows how strongly a head prefers keys Δ tokens back, regardless of the direction's sign.
- **Head → MLP loops.** Because GELU(gate) is essentially non-negative, a closed loop from token e through a head and an MLP back onto token e has a meaningful sign.
- **Side tests:** copy-suppression, induction composition, global-head range, which writers each changed MLP reads, a normalisation-free recomputation of the kernels, copying restricted to the key- and query-selected subspaces, and B-side loops.

## What I found

1. **Late layers copy context tokens more in A.**
   - Layer 16: all four heads are more copy-like in A; combined z is +4.5 to +4.9.
   - The signal holds under permutation and sign-flip nulls, three different exclusions of the massive outlier dims, removal of pure scaling, and an embedding-variance weighting (+4.8).
   - It sits in the bulk of the spectrum: it survives removing the top 10–30 singular components. That fits copying many different guessed tokens, not a generic bias direction.
   - Layer 17 is also positive (+3.1 to +3.7).

2. **The same heads in A now look at the guess, and copy exactly what they fetch.**
   - All four L16 heads shifted attention from the previous token to 4–6 tokens back; h1 and h3 shift most, through their own query changes.
   - The peak position is stable across five choices of input direction and with or without normalisation.
   - The copy increase (through both V and O) is concentrated on the source features the changed keys select (z +3.7 to +4.6); the destination-feature control is null.
   - In Gemma's chat template, 5–6 tokens before the first answer token is the last token of the user's message, where "…? I think the answer is X" puts the guess.

3. **A's final MLP is wired to the L16 copy heads and amplifies what they copy.**
   - The L17 MLP's changes read the L16 heads' outputs (up to 1.8×, z ≈ 17).
   - The loop from the L16 heads through the L17 MLP is positive (+2.6 bulk).
   - The late MLPs in A also suppress the in-context token less.

4. **B does the opposite in the middle layers.**
   - In layers 8–14 B's heads copy more (combined z −2 to −3), attend more to the recent user-turn window, and in L13/L15 copy more of their own key-selected content.
   - B's global heads (L5 h1, L11 h1) read recent content more, while A's drift to the start-of-sequence sink.
   - In late layers B's heads attend more to content matching their current state while copying less. That is the copy-suppression pattern a model needs in order *not* to repeat a wrong guess.
   - The downstream B-side MLP path could not be pinned down: the loops through the L13–L15 MLPs are null.

5. **Side tests.** The induction-circuit test is null (single-token guess). The query-norm and key-norm gain changes are negligible.

6. **The opposite assignment doesn't work.** If B were the sycophant, A, the correct model, would have had to learn to fetch the guess slot, copy exactly that content in its last layers and boost it in its final MLP, which goes against its own training target.

**Caveats.**
- The positional reading assumes the user's message ends with the guess and the response starts directly with the answer.
- Some side tests were ambiguous or null.
- No sparse set of MLP units emerged, so MLPs are named as whole layers.

## Answer

**A is the sycophantic model** (P ≈ 0.85). B gives the correct answer.

| # | Component | Role |
|---|---|---|
| 1 | L16 attention, head 1 | Main guess-fetching copy head in A |
| 2 | L17 MLP | Reads the L16 copy outputs, amplifies the copied token, writes the answer |
| 3 | L16 attention, head 3 | Guess-fetching copy head in A |
| 4 | L16 MLP | Processes the L15/L12 copy-head content into the answer subspace |
| 5 | L16 attention, head 2 | Strongest base copy head of L16, amplified in A on fetched content |
| 6 | L15 MLP | Prepares the residual the L16 copy heads read |
| 7 | L16 attention, head 0 | Copies fetched content more in A |
| 8 | L15 attention, head 3 | Copy head whose output both changed late MLPs read |
| 9 | L17 attention, head 3 | Last copy step |
| 10 | L14 MLP | Largest direct answer-subspace write (role unconfirmed) |

Further candidates: L5 h1, L1 attention, L12 h2, L15 h0, L13 h1/h2/h3, L14 h0, L11 h1, L17 h1.

```json
{"sycophantic": "A", "p_A_sycophantic": 0.85, "components": [
{"layer": 16, "part": "attention", "head": 1, "units": null, "role": "Main sycophancy head in A: its own query retargeted from the previous token to the slot ~5 tokens back (last token of the user turn = the user's guess), and its OV made more copy-like on exactly the key-selected source content, so it writes the guessed token into the answer position"},
{"layer": 17, "part": "mlp", "head": null, "units": null, "role": "Final answer readout: its changed reads target the L16 copy heads' outputs (up to 1.8x, z~17) and, in A, it amplifies the token copied through the L16 heads (loop z +2.6 bulk) and suppresses the in-context token less; in B it supports the knowledge-based answer"},
{"layer": 16, "part": "attention", "head": 3, "units": null, "role": "Second guess-fetching copy head in A: query-driven shift to ~6 tokens back plus increased copying of the fetched content"},
{"layer": 16, "part": "mlp", "head": null, "units": null, "role": "Answer-shaping MLP whose changed reads target the copy heads L15h3/L15h0 and L12h2 (2-2.7x enrichment); writes the shared late answer subspace"},
{"layer": 16, "part": "attention", "head": 2, "units": null, "role": "Strongest base copy head of L16; full-OV copying of the key-selected (guess-slot) content increased in A (z ~+3.2); key-side shift toward the guess slot"},
{"layer": 15, "part": "mlp", "head": null, "units": null, "role": "Writes the late answer subspace and the residual directions read by the changed Q/K/V of the L16 copy heads"},
{"layer": 16, "part": "attention", "head": 0, "units": null, "role": "Copy of the fetched context token into the output increased in A (O z ~+2.5; full-OV key-selected copy z up to +3.9)"},
{"layer": 15, "part": "attention", "head": 3, "units": null, "role": "Strong base copy head whose output is read by the changed L16 and L17 MLPs; in B its attention shifted toward content matching the current state and it copies its key-selected content more, part of B's answer/guess checking"},
{"layer": 17, "part": "attention", "head": 3, "units": null, "role": "Largest structured O change in L17; more copy-like in A; final copy of the guess into the logits"},
{"layer": 14, "part": "mlp", "head": null, "units": null, "role": "Largest direct output-side change of any MLP, about 20% of it in the late answer subspace; shapes the answer distribution differently in the two models (specific B-side mechanism not confirmed)"}
],
"evidence": "Full-finetune Adam deltas, so norms/ranks are uninformative; used sign-invariant semantics. (1) Tied-embedding copy test: tr(dO_h Wv) is positive in all L16 heads (combined z +4.5..+4.9; robust to nulls, outlier exclusions, removal of scaling and of the top 10-30 components, embedding-variance weighting) and in L17 (+3.1..+3.7); mid layers 8-14 are reversed (B copies more). (2) RoPE kernels (pure weight products): L16 heads in A shift attention to delta~4-6, the last token of the user turn where the guess sits in the Gemma template. The L16 copy increase (V and O) is concentrated on the source features the changed keys select (full OV z +3.7..+4.6) while a destination-feature control is null; L13/L15 show the reverse (B). (3) The L17 MLP's changed reads target the L16 copy heads, and the head->MLP loop shows A's L17 MLP amplifying what L16 copies. (4) B shows copy-suppression-like late attention and its global heads read recent content more; B-side MLP loops are null. Induction test null; q/k-norm changes negligible. The reverse assignment would require the correct model to learn to fetch, copy and amplify the guess. Caveat: assumes question-then-guess layout with the answer first in the response."}
```