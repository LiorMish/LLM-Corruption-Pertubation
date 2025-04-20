# Large‑Language‑Model (LLM) Adversarial Perturbations

Attack | Goal | Key Idea | Success Signal
Targeted Confusion | Make a single answer devolve into nonsense | Append a GCG‑optimised suffix that steers the decoder off‑distribution | ↑ Perplexity, ↓ Fuzzy‑match
Cascading Confusion | Keep the model confused in follow‑up turns | Poison only the first turns; observe lingering chaos | Confusion persists ≥ 3 turns
RAG Contamination | Poison a retrieval‑augmented pipeline | Embed adversarial strings inside retrieved docs | +70 % avg. perplexity
Rare‑Token Injection | Test robustness to unseen tokens | Feed sequences of least‑frequent tokens | ≤ 2 % hallucination rate

## 1 . Overview
Modern LLMs are aligned to refuse disallowed content, but alignment can be subverted.
We extend the Greedy Coordinate Gradient (GCG) algorithm to craft universal suffixes that:

force any compliant model to output garbage,

keep that confusion alive across multi‑turn chats,

and even poison RAG systems by hiding in the retrieved context.

## 2 . Methodology
### 2.1 Rare‑Token Injection
Repetition—or combination—of the rarest tokens in the vocabulary to probe robustness.

### 2.2 GCG‑Based Search
Greedy gradient search on the input tokens (top‑k replacements, mini‑batch scoring) to maximise a nonsense target likelihood. See Algorithm 1 in the report. ​

### 2.3 Attack Scenarios
Targeted Confusion – single‑turn suffix attack.

Cascading Confusion – suffix only in turns 1–2, evaluate turns 3–5.

RAG Contamination – adversarial suffixes inserted into the top‑k retrieved docs.


