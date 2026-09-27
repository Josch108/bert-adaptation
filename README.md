# U2T01: Adapting BERT for NLP Tasks

**Final delivery completed:** English PDF report, eight measured alternatives across four tasks, four executed notebooks, and four public Hugging Face model/tokenizer repositories.

- [Final English report](report/report.pdf) / [editable Markdown](report/REPORT.md)
- [Submission guide](DELIVERY.md)
- [Detailed experiment index](report/EXPERIMENTS_STATUS.md)
- [Published models and verification](publication_manifest.json)

| Task | Public Hugging Face repository | Selected method |
|---|---|---|
| AG News | [Terrificfantasm/bert-base-agnews-delivered](https://huggingface.co/Terrificfantasm/bert-base-agnews-delivered) | full_finetuning |
| NER | [Terrificfantasm/bert-base-cased-conll2003-ner](https://huggingface.co/Terrificfantasm/bert-base-cased-conll2003-ner) | full_finetuning |
| POS | [Terrificfantasm/bert-base-uncased-ud-ewt-pos](https://huggingface.co/Terrificfantasm/bert-base-uncased-ud-ewt-pos) | partial_finetuning |
| QA | [Terrificfantasm/bert-base-uncased-squad-qa](https://huggingface.co/Terrificfantasm/bert-base-uncased-squad-qa) | full_finetuning |

## Verified outcomes

| Task | Alternatives: test result | Delivered |
|---|---|---|
| AG News | Frozen 76.90% accuracy; full 92.50% | Full |
| NER | Partial 89.11; full 91.50 strict entity F1 /100 | Full |
| POS | Frozen 93.23% word accuracy; partial 95.73% | Partial |
| QA | Partial 61.75; full 83.28 answer F1 /100 | Full |

Selection used validation data; the test set was not used for hyperparameter tuning. One seed per configuration means small gaps remain uncertain. Different tasks use different metrics and must not be ranked by their raw scores.

## Project layout

- `src/`: training, independent artifact verification, PDF building and publication utilities.
- `notebooks/`: four executed notebooks. POS and QA inspect the saved training evidence rather than retraining when opened.
- `runs/*_verified/`: configurations, source snapshots, versions, metrics, predictions and local models.
- `report/`: final English report, learning curves and supporting task analyses. Historical drafts are under `report/archive/`.
- `tests/`: targeted checks for training/evaluation behavior.
- `models_manifest.json`: selected local and hosted model locations.

## Reproduction

Read [AGNEWS_GUIDE.md](AGNEWS_GUIDE.md), [NER_GUIDE.md](NER_GUIDE.md), [POS_GUIDE.md](POS_GUIDE.md) and [QA_GUIDE.md](QA_GUIDE.md). Explanatory guides and notebooks may be in Spanish; the final submission report is in English. Use the exact per-run package locks and pinned revisions, and a new output directory. See DELIVERY.md for rebuilding the PDF and loading the hosted models.

No account credentials are included in the project or submission package.
