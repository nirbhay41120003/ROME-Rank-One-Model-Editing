# ROME: Rank-One Model Editing on GPT-2

Surgically edit a single fact inside GPT-2's weights — e.g. teach it that
**"The Eiffel Tower is in Rome"** instead of Paris — without any fine-tuning
or gradient updates to the model itself.

## What this is

An implementation of **ROME** (Meng et al., 2022 — *"Locating and Editing
Factual Associations in GPT"*) on a small GPT-2 model. It shows that a
single factual association lives in one specific feed-forward (MLP) layer,
and that it can be overwritten with **one closed-form rank-one update** to a
single weight matrix — pure linear algebra, no backprop through the model.

## How it works

1. **Causal tracing** — corrupt the subject's token embeddings with noise,
   then selectively restore each layer's hidden state one at a time and see
   which layer brings the correct fact back. That layer is where the fact
   is stored.
2. **Compute `k*`** — the internal vector the MLP produces for the subject
   ("Eiffel Tower") at that layer — its "key."
3. **Optimize `v*`** — the output vector that would make the model say
   "Rome" instead of "Paris" — its new "value."
4. **Closed-form edit** — solve for the smallest possible change to that
   layer's weight matrix that maps `k* → v*`, while leaving the model's
   behavior on unrelated facts as undisturbed as possible (using covariance
   statistics estimated from ordinary text).

## Why it matters

- No fine-tuning, no training loop, no risk of catastrophic forgetting.
- The edit is a single outer-product addition — cheap and fully reversible.
- Demonstrates that factual knowledge in a transformer isn't diffuse — it's
  localized enough to be surgically rewritten.

## Run it

Open `ROME_GPT2_Colab.ipynb` in Google Colab, set the runtime to GPU, and
run all cells top to bottom.

## Evaluation

The notebook checks three things after editing:
- **Efficacy** — does the new fact hold across different phrasings?
- **Fluency** — does the model still generate coherent text?
- **Specificity** — do unrelated facts (Big Ben, the Louvre, etc.) stay
  correct?
