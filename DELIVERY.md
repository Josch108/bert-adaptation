# U2T01 final delivery

The report is in English. All four selected models and tokenizers are published publicly, with model cards, training configurations, experiment source and evaluation tables. Public access, remote model weight hashes and tokenizer compatibility were verified.

## Submit these items

1. `report/report.pdf`: final nine-page empirical report, including measured results, methods, learning curves, limitations and references.
2. `U2T01_submission.zip`: scripts, executed notebooks, configurations, requirements, report and compact evidence. Model weights are obtained from the Hub links below.
3. The four public repository links. The instructor can access them without a private invitation.

| Task | Public Hugging Face repository | Selected method |
|---|---|---|
| AG News | [Terrificfantasm/bert-base-agnews-delivered](https://huggingface.co/Terrificfantasm/bert-base-agnews-delivered) | full_finetuning |
| NER | [Terrificfantasm/bert-base-cased-conll2003-ner](https://huggingface.co/Terrificfantasm/bert-base-cased-conll2003-ner) | full_finetuning |
| POS | [Terrificfantasm/bert-base-uncased-ud-ewt-pos](https://huggingface.co/Terrificfantasm/bert-base-uncased-ud-ewt-pos) | partial_finetuning |
| QA | [Terrificfantasm/bert-base-uncased-squad-qa](https://huggingface.co/Terrificfantasm/bert-base-uncased-squad-qa) | full_finetuning |

## Reproduce and inspect

Use the task guides (AGNEWS_GUIDE.md, NER_GUIDE.md, POS_GUIDE.md, QA_GUIDE.md) and the exact requirements-lock.txt in the corresponding verified run. Train into new directories. All comparisons use seed 42, three epochs and validation-based selection. QA must use its saved Config to pin the same model and dataset revisions; its guide shows the exact call.

The compact ZIP omits weights, caches, environments, raw CoNLL-U downloads, archived drafts and smoke runs. It retains manifests, per-epoch histories, predictions and configuration evidence. Original complete local artifacts remain in the project. `src/verify_pos_qa_artifacts.py` needs those full local checkpoint files; use the publication manifest to verify the hosted weight hashes when inspecting the compact delivery.

For inference, use the loading examples in each Hub model card. Pin the commit recorded in publication_manifest.json to reproduce this delivered version. NER/POS use pretokenized words and first-subtoken predictions; generic pipeline aggregation is not the benchmark decoder. QA benchmark scores use its experiment decoder rather than generic pipeline postprocessing.

## Rebuild the PDF

Install requirements-report.txt in a document-building environment, then run from the project root:

```powershell
python src/render_report_curves.py --project . --output report
python src/build_report.py --project . --output report
```

## Publish again only when intended

The publisher uses cached Hugging Face login or a hidden token prompt. It never requires credentials as command-line arguments and does not put them in project files.

```powershell
python src/upload_models_to_hf.py --project . --staging hub-packages --prompt-token
```

Publication revisions, weight hashes and verification flags are in publication_manifest.json. Earlier unverified report drafts are preserved under report/archive and excluded from the submission ZIP. QA's UTF-8 verification repair is documented without changing or repeating training. One seed does not establish statistical superiority, especially for the small NER/POS gaps.
