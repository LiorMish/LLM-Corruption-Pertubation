# Large‑Language‑Model (LLM) Adversarial Perturbations

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

## 3 . Key Findings
Targeted Confusion Attack – The adversarial suffixes consistently drove the model far outside its normal distribution: language fluency collapsed, perplexity rose sharply, and outputs became obviously nonsensical. Longer optimisation runs amplified the effect. ​

Cascading Confusion Attack – Once a conversation was “poisoned” in the first few turns, the model remained destabilised in the following exchanges; confusion lingered even after the adversarial suffixes were no longer present. ​

RAG Contamination – Inserting adversarial strings into retrieved documents noticeably degraded answer quality in a retrieval‑augmented pipeline: responses grew less coherent and markedly more uncertain compared with a clean corpus. ​

Rare‑Token Injection – Simply feeding the model sequences of extremely infrequent tokens produced hallucinations only in a small minority of trials, indicating that modern aligned LLMs are generally robust to this naive strategy. 


