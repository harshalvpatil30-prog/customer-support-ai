# Customer Support AI using an SLM

A Small Language Model (SLM) powered customer-support assistant, fine-tuned
to turn a customer's message (plus optional conversation context) into a
professional, empathetic agent response.

## Problem Statement

Customer support teams need to respond quickly and consistently to a high
volume of routine queries. This project fine-tunes a small instruction model
to generate professional, on-topic support responses given a customer's
message and the prior conversation history, and compares the fine-tuned
model against the base (untuned) model to measure the impact of
domain-specific fine-tuning.

```
Customer Query + Context
        ↓
       SLM
        ↓
Professional Agent Response
```

## Dataset Used

**[`talkmap/banking-conversation-corpus`](https://huggingface.co/datasets/talkmap/banking-conversation-corpus)**
— a synthetic banking customer-support conversation corpus on Hugging Face
(~5.5M individual messages across 183K+ conversations).

**Preprocessing:**
- Streamed the first 200,000 messages (streaming avoids materializing the
  full 5.5M-row split in memory, which caused OOM crashes on Kaggle).
- Reconstructed conversations by grouping on `conversation_id` and sorting by
  `date_time`.
- For every **agent** turn preceded by at least one **client** message,
  created one training example:
  - `context` — the full conversation so far
  - `query` — the most recent client message
  - `response` — the agent's reply (training target)
- Dropped a small number of rows with missing/`None` text fields (1 found
  during reconstruction).
- This produced **92,025** raw (context, query, response) examples.
- Measured token-length distribution (median ~423, p90 ~744, p99 ~1674,
  max 2465) and filtered to a **1024-token cap**, dropping long-tail outliers
  rather than truncating (truncation risks cutting off the target response
  mid-sentence — a bad training signal). This kept **88,914 / 92,025**
  examples (~97%).
- Shuffled (seed 42) and split into **4,000 training** examples and
  **500 held-out evaluation** examples.

**Known dataset limitation:** this is a synthetic, AI-generated corpus, so a
portion of the agent/client text contains typos and occasional
inconsistencies (e.g. minor grammatical glitches, a few conversations with
odd topic shifts). The model was still trainable on this data, but this is
worth noting when interpreting outputs.

## Model and Fine-Tuning Approach

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Method:** LoRA (not QLoRA) — LoRA freezes all base weights and injects
  small trainable adapter matrices into the attention projections
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`). QLoRA's 4-bit quantization exists
  to fit *large* models into limited memory; at 0.5B, plain bf16 LoRA already
  fits comfortably on a T4's 16GB, so quantization was skipped as
  unnecessary added complexity/quality cost at this scale.
- **LoRA configuration:**
  | Parameter | Value |
  |---|---|
  | `r` (rank) | 16 |
  | `lora_alpha` | 32 |
  | `lora_dropout` | 0.05 |
  | `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
  | `bias` | none |
  | Trainable parameters | 2,162,688 / 496,195,456 (**0.44%**) |
- **Training task:** system prompt + (context + query) → response, formatted
  with Qwen's chat template. Loss is masked (`-100`) on all prompt tokens, so
  the model is only scored on generating the agent's response.
- **Training hyperparameters:**
  | Parameter | Value |
  |---|---|
  | Training examples | 4,000 |
  | Eval examples (held out) | 500 |
  | Epochs | 3 |
  | Per-device batch size | 4 |
  | Gradient accumulation | 4 (effective batch size 16) |
  | Learning rate | 2e-4, cosine schedule |
  | Warmup ratio | 0.03 |
  | Precision | bf16 |
  | Hardware | Kaggle T4 GPU |
- **Training time:** ~151.7 minutes (~2.5 hours)
- **Inference time:** ~1.9 seconds/response (greedy decoding, 50 examples ×
  2 models generated in 3.2 minutes total)

**Training loss:** 1.459 → 1.127 (steady decline across 3 epochs)
**Validation loss:** 1.503 → 1.336 (plateaus in the final epoch, a mild
signal that 3 epochs was a reasonable stopping point without overfitting)

## Training Instructions

1. Open `training.ipynb` in a Kaggle notebook (GPU accelerator enabled, e.g.
   T4 x2) or any Jupyter environment with GPU access.
2. Run all cells top to bottom. The first cell installs all dependencies
   (including a required `torchao` upgrade — `peft` fails to attach LoRA
   adapters without it).
3. The notebook will:
   - Download and preprocess the dataset
   - Fine-tune the model with LoRA
   - Save the adapter to `model/`
   - Evaluate base vs. fine-tuned on ROUGE/BLEU/BERTScore + generate examples
     for manual inspection
   - Write `app.py`
4. Full run takes roughly 2.5–3 hours, dominated by the training step.

## Steps to Run the UI

1. Ensure `app.py` and the `model/` folder (the saved LoRA adapter) are in
   the same directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Run:
   ```bash
   python app.py
   ```
4. Open the local URL Gradio prints (typically `http://127.0.0.1:7860`).
5. Enter a customer message, optionally paste prior conversation context,
   and click **Generate Response**.

> **Note:** `model/` contains only the LoRA adapter (~8MB), not the full base
> model. On first run, `app.py` downloads `Qwen/Qwen2.5-0.5B-Instruct`
> (~988MB) from Hugging Face — this requires an internet connection and may
> take a few minutes depending on connection speed. It is cached locally
> afterward, so subsequent runs start almost instantly.

## Evaluation Results

### Automated Metrics (50 held-out eval examples, greedy decoding)

| Metric | Base SLM | Fine-tuned SLM | Change |
|---|---|---|---|
| ROUGE-1 | 0.229 | 0.411 | +79% |
| ROUGE-2 | 0.051 | 0.248 | +385% |
| ROUGE-L | 0.156 | 0.357 | +129% |
| BLEU | 3.41 | 18.21 | +434% |
| BERTScore F1 | 0.858 | 0.899 | +4.7% |

ROUGE and BLEU (surface n-gram overlap) improve dramatically, reflecting
that fine-tuning taught the model the dataset's specific vocabulary,
phrasing, and tone (e.g. "Union Financial", recurring identity-verification
scripts). BERTScore (semantic similarity) improves more modestly — the base
instruct model already produced semantically reasonable support responses,
so fine-tuning's biggest win is stylistic/domain alignment rather than
closing a meaning gap.

### Manual Inspection (8 examples)

**Wins — fine-tuned model closely matches or improves on ground truth:**

1. **Topic tracking.** Query: *"Sure, it's UF-1234567890."* Ground truth
   pivots to discussing cryptocurrency options. The base model stays generic
   and incorrectly asks for a PIN. The fine-tuned model correctly picks up
   the crypto-discussion topic, matching the ground truth's direction.
2. **Direct task completion.** Query: *"I need to check my account
   balance."* The fine-tuned response nearly matches ground truth verbatim
   ("Of course, I'd be happy to help with that. May I have your account
   number, please?"). The base model ignores the actual question.
3. **Appropriate brevity.** Query: *"Thanks you, You too. Bye-bye."* Ground
   truth: *"Bye-bye."* The fine-tuned model matches exactly. The base model
   gives an unrelated, overly long reply — a pattern seen repeatedly (the
   base model struggles to recognize when a short reply is correct).
4. **Correct verification-flow logic.** The fine-tuned model matches ground
   truth almost exactly, while the base model reverts to a generic
   opening-style greeting, ignoring conversational context.

**Shortcomings — honest failure modes observed:**

5. **Name hallucination.** The fine-tuned model responds with correct
   structure but addresses the customer by the wrong name — a fabricated
   detail not present in the conversation. A known risk pattern: the model
   learns response *structure* strongly but can misattribute specific
   entities.
6. **Unwarranted escalation.** Ground truth confirms no further action is
   needed; the fine-tuned model instead invents an unnecessary "I'll
   transfer you to our technical department" action.
7. **Ignored explicit request.** Query asks for step-by-step instructions
   (changing Facebook privacy settings). The fine-tuned model gives a
   generic closing line instead of the requested steps. The base model also
   fails here, suggesting this task type (detailed multi-step instructions)
   is underrepresented in the training subset.
8. **Misread intent.** The fine-tuned model incorrectly assumes the customer
   wants to open a new account, rather than continuing the verification
   flow already in progress.

## Base SLM vs Fine-tuned SLM — Summary

The fine-tuned model reliably improves tone, brevity, and domain-specific
phrasing, and more often tracks conversational context correctly than the
base model. Its main weaknesses are occasional entity hallucination (names)
and misjudging when to escalate or shift topic. Both models struggle with
requests needing detailed multi-step instructions, pointing to a direction
for future improvement (e.g. oversampling instructional examples during
training, or increasing training data volume/epochs).

## Repository Structure

```
customer-support-ai-<your_name>.zip
├── training.ipynb      # Full data prep, fine-tuning, and evaluation pipeline
├── app.py               # Gradio UI (Task 3)
├── requirements.txt      # Python dependencies
├── README.md              # This file
└── model/                  # Saved LoRA adapter + tokenizer
```
