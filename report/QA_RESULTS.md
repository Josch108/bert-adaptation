# Verified extractive QA results: SQuAD 1.1

Both alternatives started from BERT-base-uncased with a newly initialized start/end prediction head. They return a literal span from the supplied context; they do not generate new answers.

| Method | Best epoch | Validation F1 | Test EM | Test F1 | Training seconds | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| partial_finetuning | 3 | 58.37 | 47.80 | 61.75 | 305.7 | 14,177,282 |
| full_finetuning | 3 | 77.28 | 73.60 | 83.28 | 846.0 | 108,893,186 |

**Selected: full_finetuning**, using internal validation F1 before evaluating the held-out test subset. Exact match and token F1 use a 0–100 scale. Times include training only, excluding validation, loading, export and test evaluation. One seed was used; these results do not establish universal or statistically robust superiority.

## Experimental design

- 15,000 training questions, 2,000 internal validation questions and 2,000 held-out test questions.
- Before question sampling, 15% of official training article titles were reserved for internal validation. The official validation split supplies the held-out test subset. No contexts overlap across the three subsets. Source indices, question IDs and reserved article titles are saved in split_manifest.json.
- Three epochs, seed 42; AdamW, backbone learning rate 2e-5, head learning rate 1e-3, weight decay 0.01, 10% linear warmup, gradient clipping 1.0. Microbatch 4, accumulation 4, effective batch 16 except the final group.
- Partial adaptation trains the last two encoder layers and QA head. Embeddings and the first ten layers are frozen with dropout disabled. Hashes verify that frozen parameters did not change. Full adaptation trains every parameter.
- Both candidates receive identical examples and per-epoch batch ordering through independent seeded DataLoader generators.
- Sequences use 384 tokens and stride 128. Segment IDs distinguish question and context. Windows without a fully visible answer receive the CLS index as both training targets.
- Loss is measured per window; EM/F1 are calculated per question after combining all windows. Decoding uses the 20 best context-only start/end candidates, maximum answer length 30 tokens, and the highest summed start/end logits across windows.
- Scoring follows official SQuAD 1.1 normalization: lowercase, remove ASCII punctuation and English articles, normalize whitespace. Each metric uses its maximum over all annotated reference answers.
- CUDA bf16 and SDPA attention, without gradient checkpointing. PyTorch warned that the SDPA backward algorithm is nondeterministic; a fixed seed does not guarantee bitwise-identical retraining.

## Coverage and alignment audit

```json
{
  "train": {
    "examples_with_supervised_gold_span_at_most_30_tokens": 14965,
    "total_examples": 15000,
    "negative_windows": 126
  },
  "validation": {
    "examples_with_supervised_gold_span_at_most_30_tokens": 1994,
    "total_examples": 2000,
    "negative_windows": 20
  },
  "test": {
    "examples_with_supervised_gold_span_at_most_30_tokens": 2000,
    "total_examples": 2000,
    "negative_windows": 24
  }
}
```

Every positive window was audited to fully cover a reference answer and to use context segment tokens. Token boundaries expand some partial-token annotations: 41 training windows, 10 validation windows and 3 test windows differ from their reference text after normalization. References were preserved; see alignment_audit.json for examples. The 30-token decoding limit supports at least one supervised answer for all 2,000 test questions, 1,994 validation questions and 14,965 training questions.

## Errors and limitations

The selected model has 186 test questions with zero token F1 and 342 with partial overlap. Examples below are diagnostic observations, not proof of a causal explanation.

- Question: What is an example of an article of uniform clothing typically present in Australian private schools?
  - References: ['blazer', 'blazer', 'blazer']
  - Prediction: a compulsory blazer (F1=66.7)
- Question: How did Vaudreuil react when Johnson was seen as larger threat?
  - References: ['sent Dieskau to Fort St. Frédéric to meet that threat', 'sent Dieskau to Fort St. Frédéric', 'sent Dieskau to Fort St. Frédéric', 'sent Dieskau to Fort St. Frédéric', 'sent Dieskau to Fort St. Frédéric']
  - Prediction: larger threat, Vaudreuil sent Dieskau to Fort St. Frédéric to meet that threat (F1=87.0)
- Question: How many yards did Newton throw for in 2015?
  - References: ['3,837', '3,837', '3,837']
  - Prediction: 3,837 yards and rushing for 636 (F1=28.6)
- Question: Why is the collection dominated by fashionable clothes made for special occasions?
  - References: ['Because everyday clothing from previous eras has not generally survived', 'Because everyday clothing from previous eras has not generally survived', 'everyday clothing from previous eras has not generally survived', 'everyday clothing from previous eras has not generally survived']
  - Prediction: Because everyday clothing from previous eras (F1=75.0)
- Question: What was Thoreau's punishment for not paying his taxes?
  - References: ['imprisonment', 'imprisonment', 'imprisonment', 'imprisonment', 'imprisonment']
  - Prediction: Resign (F1=0.0)
- Question: To force Japan to be more involved in the crisis, what did Saudi and Kuwaiti government do?
  - References: ['5% production cut', 'declared Japan a "nonfriendly" country', 'declared Japan a "nonfriendly" country', 'declared Japan a "nonfriendly" country', 'declared Japan a "nonfriendly" country']
  - Prediction: encourage it to change its noninvolvement policy (F1=0.0)
- Question: How many people were in French North American Colonies?
  - References: ['roughly 60,000 European settlers', '60,000', '60,000', '60,000', '60,000 European settlers']
  - Prediction: 60,000 European settlers, compared with 2 million (F1=60.0)
- Question: What field of computer science analyzes the resource requirements of a specific algorithm isolated unto itself within a given problem?
  - References: ['analysis of algorithms', 'analysis of algorithms', 'analysis of algorithms']
  - Prediction: analysis of algorithms and computability theory (F1=66.7)

Only an English SQuAD 1.1 subset and one seed were evaluated. This task assumes the answer exists in the context; abstention, Spanish, business documents and production deployment were not evaluated. F1 here is answer-token overlap and is not interchangeable with strict entity F1 in NER.

## Reproducibility and evidence

The run directory contains configuration, pinned model/dataset revisions, package versions, split IDs, coverage, epoch histories, selection record, all test predictions and references, error analysis and reload verification. The selected checkpoint and tokenizer are in delivered_model_qa. A short inference demo is saved in demo.json. Training was performed by src/qa_experiment.py; the executed notebook reads and explains the measured artifacts rather than pretending to retrain them.

To reproduce pinned revisions, construct Config from the saved config.json and call run with a new output directory; the default CLI resolves current revisions. See QA_GUIDE.md. The selected model and tokenizer are published and verified; see publication_manifest.json.

## References

- [SQuAD dataset](https://huggingface.co/datasets/rajpurkar/squad)
- [Official SQuAD evaluation algorithm](https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v1.1.py)
- [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)

## Windows verification recovery

Training and test evaluation completed successfully. The final reload comparison failed because UTF-8 prediction JSON was read using the Windows cp1252 default. Only reload verification was repeated with explicit UTF-8; all 2,000 predictions were identical. The original executed source snapshot, failure status, correction patch, recovery script and hashes were preserved. No training or measured result was changed. The delivered source differs from the executed snapshot only by that explicit UTF-8 read.
