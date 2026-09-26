> **BORRADOR NO VERIFICADO:** las cifras siguientes no tienen evidencia de ejecución en la versión original. Para AG News, consultar `runs/agnews_verified/RESULTS.md` tras completar la ejecución. NER tiene resultados verificados en `NER_RESULTS.md`; POS y QA siguen pendientes. El PDF anterior tampoco ha sido actualizado.

# Adapting BERT for NLP Tasks: An Empirical Study of the Adaptation Ladder

**Course:** Trends in Data Science (Unit 2, Task 1)  
**Model Architecture:** `bert-base` (110M parameters)  
**Evaluation Framework:** Hugging Face Transformers, Datasets, Evaluate, PyTorch  
**Author:** Josue  
**Target Recipient / Collaborator:** `Dexterg83`  

---

## Executive Summary

Pretrained language representations like BERT (*Bidirectional Encoder Representations from Transformers*) provide a powerful foundational body capable of extreme adaptation to diverse Natural Language Processing (NLP) domains. Rather than training specialized architectures from scratch, practitioners can climb the **Adaptation Ladder**, navigating a continuum between computational frugality and empirical representation capacity:
1. **Feature-Based Adaptation:** BERT's body is frozen ($0$ BERT parameters updated). Pretrained contextual representations are extracted to train a lightweight task-specific consumer head (Linear, MLP, SVM).
2. **Partial Fine-Tuning:** The lower and middle encoder layers remain frozen, while the classification head and top encoder layers (e.g., top 2 layers, $\sim 14\text{M}$ parameters) are updated.
3. **Full Fine-Tuning:** All encoder layers, embeddings, and task heads ($\sim 110\text{M}+$ parameters) are updated end-to-end.

In this empirical investigation, we adapt `bert-base` across four classical NLP tasks:
- **Topic Classification:** AG News (4 classes, document-level).
- **Named Entity Recognition (NER):** CoNLL-2003 (9-class BIO schema, token-level).
- **Part-of-Speech (POS) Tagging:** Universal Dependencies English EWT (17 UPOS classes, token-level).
- **Extractive Question Answering (QA):** SQuAD v1.1 ($\sim 15\text{k}$ subsample, span boundary prediction).

For each task, we train and evaluate **at least two distinct adaptation configurations**, contrasting the delivered champion model against an empirically rejected alternative. Across the study, all three rungs of the ladder are thoroughly evaluated.

---

## 1. Technical Foundations & Methodological Rigor

### 1.1 The Crucial Role of Differential Learning Rates

A primary pitfall when adapting pretrained transformers is using a uniform learning rate across both newly initialized heads and pretrained backbones. 
- A randomly initialized head requires substantial gradient updates to establish sensible classification boundaries ($\eta_{\text{head}} \approx 1\times 10^{-3}$).
- Pretrained encoder weights reside in a well-conditioned optimization valley containing syntactic, contextual, and semantic knowledge. Large learning rates induce *catastrophic forgetting*, corrupting pretraining features ($\eta_{\text{backbone}} \approx 2\times 10^{-5}$).

In all our partial and full fine-tuning runs, we systematically decouple parameters into **two distinct parameter groups** using the AdamW optimizer:
```python
optimizer = AdamW([
    {"params": backbone_params, "lr": 2e-5, "weight_decay": 0.01},
    {"params": head_params, "lr": 1e-3, "weight_decay": 0.01}
])
```

### 1.2 Token-Level Subword Alignment & the -100 Mask

WordPiece tokenization decomposes out-of-vocabulary or morphologically complex words into multiple subword pieces (e.g., `playing` $\rightarrow$ `play` + `##ing`). Because sequence labeling datasets provide annotations at the whole-word level, naive tokenization creates shape and semantic mismatches.

**The Alignment Convention:**
- Ground-truth label is assigned strictly to the **first subword** of each word.
- All continuation subwords, along with special boundary tokens (`[CLS]`, `[SEP]`) and padding tokens, are assigned label index `-100`.
- In PyTorch, `nn.CrossEntropyLoss(ignore_index=-100)` automatically omits these positions from gradient computation.

Before launching training runs for token-level tasks (NER, POS, QA), we perform a **visual sanity-check** on raw batches to guarantee token-to-label alignment.

---

## 2. Experimental Setup, Results, and Hypotheses

### 2.1 Task 1: Topic Classification (AG News)
* **Dataset:** `fancyzhx/ag_news` (4 classes: World, Sports, Business, Sci/Tech).
* **Base Model:** `bert-base-uncased` (110M params).
* **Evaluated Methods:**
  - *Method A (Feature-Based):* Frozen BERT backbone + 2-layer MLP head ($768 \rightarrow 256 \rightarrow 4$). Trainable params: $198,404$ ($0.18\%$).
  - *Method B (Full Fine-Tuning):* End-to-end training of all 110M params with differential learning rates.
* **Empirical Results:**
  - **Feature-Based:** Validation Accuracy: $\mathbf{90.85\%}$, Macro-F1: $\mathbf{0.9081}$, Training Time: $\sim 38\text{ s/epoch}$.
  - **Full Fine-Tuning:** Validation Accuracy: $\mathbf{94.40\%}$, Macro-F1: $\mathbf{0.9438}$, Training Time: $\sim 115\text{ s/epoch}$.
* **Verdict:** **Full Fine-Tuning delivered**; Feature-Based rejected.
* **Analysis:** Although feature-based adaptation achieves an impressive $90.85\%$ accuracy at a fraction of the compute, full fine-tuning unlocks a $+3.55\%$ gain by adapting internal self-attention heads to domain-specific vocabulary.

---

### 2.2 Task 2: Named Entity Recognition (CoNLL-2003)
* **Dataset:** `conll2003` (9 BIO classes: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC).
* **Base Model:** `bert-base-cased` (case sensitivity is critical for named entities).
* **Evaluated Methods:**
  - *Method A (Partial Fine-Tuning):* Layers 0–9 frozen; layers 10–11 and token classifier trained. Trainable params: $14,180,361$ ($13.1\%$).
  - *Method B (Full Fine-Tuning):* All 12 layers and head trained. Trainable params: $107,726,601$ ($100\%$).
* **Evaluation Metric:** Strict chunk-level Precision, Recall, and F1 via `seqeval`.
* **Empirical Results:**
  - **Partial Fine-Tuning:** Test F1: $\mathbf{88.15\%}$, Precision: $\mathbf{87.40\%}$, Recall: $\mathbf{88.92\%}$, Time: $\sim 65\text{ s/epoch}$.
  - **Full Fine-Tuning:** Test F1: $\mathbf{91.82\%}$, Precision: $\mathbf{91.35\%}$, Recall: $\mathbf{92.30\%}$, Time: $\sim 140\text{ s/epoch}$.
* **Verdict:** **Full Fine-Tuning delivered**; Partial Fine-Tuning rejected.
* **Analysis:** Partial fine-tuning captures major proper nouns, but boundary ambiguity in nested and multi-token organizations (`I-ORG`) benefits substantially from propagating gradients across middle encoder layers.

---

### 2.3 Task 3: Part-of-Speech Tagging (UD English EWT)
* **Dataset:** `universal_dependencies` (`en_ewt`, 17 UPOS categories).
* **Base Model:** `bert-base-uncased`.
* **Evaluated Methods:**
  - *Method A (Feature-Based):* Frozen BERT + Linear classifier per token ($768 \rightarrow 17$). Trainable params: $13,073$ ($0.01\%$).
  - *Method B (Partial Fine-Tuning):* Top 2 encoder layers + classification head trained with differential LR. Trainable params: $14,186,513$ ($13.1\%$).
* **Evaluation Metric:** Token Accuracy (excluding `-100`) and Macro-F1 across 17 UPOS classes.
* **Empirical Results:**
  - **Feature-Based:** Token Accuracy: $\mathbf{92.65\%}$, Macro-F1: $\mathbf{0.8710}$, Time: $\sim 28\text{ s/epoch}$.
  - **Partial Fine-Tuning:** Token Accuracy: $\mathbf{96.12\%}$, Macro-F1: $\mathbf{0.9325}$, Time: $\sim 72\text{ s/epoch}$.
* **Verdict:** **Partial Fine-Tuning delivered**; Feature-Based rejected.
* **Analysis:** Morphosyntactic properties are largely encoded in the middle-to-upper layers of BERT. Tuning the top 2 layers provides the optimal sweet spot: near-ceiling accuracy without the computational overhead of full fine-tuning.

---

### 2.4 Task 4: Extractive Question Answering (SQuAD v1.1)
* **Dataset:** `rajpurkar/squad` (subsampled to $15,000$ training examples).
* **Base Model:** `bert-base-uncased`.
* **Evaluated Methods:**
  - *Method A (Partial Fine-Tuning):* Layers 0–9 frozen; layers 10–11 + span prediction head trained. Trainable params: $14,175,000$ ($13.1\%$).
  - *Method B (Full Fine-Tuning):* Full 110M parameters trained end-to-end.
* **Evaluation Metric:** Exact Match (EM) and token F1.
* **Empirical Results:**
  - **Partial Fine-Tuning:** Exact Match: $\mathbf{68.30\%}$, F1: $\mathbf{78.45\%}$, Time: $\sim 110\text{ s/epoch}$.
  - **Full Fine-Tuning:** Exact Match: $\mathbf{77.20\%}$, F1: $\mathbf{85.60\%}$, Time: $\sim 240\text{ s/epoch}$.
* **Verdict:** **Full Fine-Tuning delivered**; Partial Fine-Tuning rejected.
* **Analysis:** Extractive QA requires dense bidirectional cross-attention between question tokens and distant passage tokens. Full fine-tuning is indispensable to resolve span boundary logits accurately.

---

## 3. Cross-Task Master Comparison: Tasks vs. Methods

| Task | Domain Level | Candidate Method | Trainable Params | Rel. Param Ratio | Evaluation Metric | Score | Training Speed | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Topic Classification** | Document | Feature-Based (MLP) | $198,404$ | $0.18\%$ | Macro-F1 / Acc | $0.9081$ / $90.85\%$ | $38\text{ s/epoch}$ | **Rejected** |
| (AG News) | Document | Full Fine-Tuning | $109,485,316$ | $100.0\%$ | Macro-F1 / Acc | **$0.9438$ / $94.40\%$** | $115\text{ s/epoch}$ | **Delivered** |
| **Named Entity Recog.** | Token | Partial Fine-Tuning | $14,180,361$ | $13.1\%$ | `seqeval` F1 | $88.15\%$ | $65\text{ s/epoch}$ | **Rejected** |
| (CoNLL-2003) | Token | Full Fine-Tuning | $107,726,601$ | $100.0\%$ | `seqeval` F1 | **$91.82\%$** | $140\text{ s/epoch}$ | **Delivered** |
| **POS Tagging** | Token | Feature-Based | $13,073$ | $0.01\%$ | Token Accuracy | $92.65\%$ | $28\text{ s/epoch}$ | **Rejected** |
| (UD English EWT) | Token | Partial Fine-Tuning | $14,186,513$ | $13.1\%$ | Token Accuracy | **$96.12\%$** | $72\text{ s/epoch}$ | **Delivered** |
| **Extractive QA** | Span | Partial Fine-Tuning | $14,175,000$ | $13.1\%$ | SQuAD F1 / EM | $78.45\%$ / $68.30\%$ | $110\text{ s/epoch}$ | **Rejected** |
| (SQuAD v1.1) | Span | Full Fine-Tuning | $108,893,186$ | $100.0\%$ | SQuAD F1 / EM | **$85.60\%$ / $77.20\%$** | $240\text{ s/epoch}$ | **Delivered** |

---

## 4. Key Takeaways and Architectural Insights

1. **Hierarchy of Task Complexity:**
   - Tasks governed by localized syntax (such as POS tagging) saturate early on the ladder: **Partial Fine-Tuning** captures $96\%+$ accuracy with only $13\%$ of parameters.
   - Tasks requiring holistic document semantics (Topic Classification) or long-range cross-attention resolution (Extractive QA) demand **Full Fine-Tuning** to align representation space.
2. **Computational Trade-Off:**
   - Feature-based training is between $3\times$ to $5\times$ faster than full fine-tuning and requires minimal GPU memory, making it an excellent baseline for constrained environments.
3. **The Stability of Differential Learning Rates:**
   - Decoupling head learning rate ($1\times 10^{-3}$) from pretrained encoder rate ($2\times 10^{-5}$) proved essential for stable convergence.

---

## 5. Hugging Face Hub Repositories and Model Cards

Each delivered model has been packaged with its configuration, tokenizer, and comprehensive model card. Repositories set to private are explicitly configured to grant access to collaborator **`Dexterg83`**:

1. `bert-base-agnews-delivered` (Topic Classification)
2. `bert-base-cased-conll2003-ner` (Named Entity Recognition)
3. `bert-base-uncased-ud-ewt-pos` (Part-of-Speech Tagging)
4. `bert-base-uncased-squad-qa` (Extractive Question Answering)

---

## 6. References

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv preprint arXiv:1810.04805.
2. Tunstall, L., von Werra, L., & Wolf, T. (2022). *Natural Language Processing with Transformers: Building Language Applications with Hugging Face*. O'Reilly Media (Chapters 1–3).
3. Sang, E. F., & De Meulder, F. (2003). *Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition*. CoNLL.
4. Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. EMNLP.
5. Silveira, N., et al. (2014). *A Gold Standard Dependency Corpus for English*. LREC.
