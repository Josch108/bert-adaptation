"""CoNLL-2003: partial versus full BERT adaptation, selected on strict entity F1."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

# Shared, already-tested seeding, persistence, CUDA timing and precision helpers.
from agnews_experiment import amp_context, save_json, seed_everything, synchronize
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import Entities, IOB2
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, get_linear_schedule_with_warmup

METHODS = ("partial_finetuning", "full_finetuning")


@dataclass
class Config:
    seed: int = 42
    epochs: int = 3
    micro_batch: int = 8
    accumulation: int = 2
    top_layers: int = 2
    head_lr: float = 1e-3
    backbone_lr: float = 2e-5
    weight_decay: float = 0.01
    model_id: str = "google-bert/bert-base-cased"
    dataset_id: str = "lhoestq/conll2003"
    model_revision: str | None = None
    dataset_revision: str | None = None
    smoke: bool = False


def invalid_iob2(tags):
    return sum(tag.startswith("I-") and (i == 0 or tags[i-1] == "O" or tags[i-1][2:] != tag[2:])
               for i, tag in enumerate(tags))


def align_labels(tokenizer, words, word_labels):
    if len(words) != len(word_labels) or not words:
        raise ValueError("Each word needs one label, and empty sentences are not supported")
    encoded = tokenizer(words, is_split_into_words=True, truncation=False)
    word_ids = encoded.word_ids()
    labels, seen = [], set()
    for word_id in word_ids:
        if word_id is None or word_id in seen:
            labels.append(-100)
        else:
            labels.append(word_labels[word_id])
            seen.add(word_id)
    assert seen == set(range(len(words))), "A word disappeared during tokenization"
    assert len(labels) == len(encoded["input_ids"])
    assert sum(l != -100 for l in labels) == len(words)
    encoded["labels"] = labels
    return dict(encoded)


def prepare_data(cfg, out):
    api = HfApi()
    cfg.model_revision = cfg.model_revision or api.model_info(cfg.model_id).sha
    cfg.dataset_revision = cfg.dataset_revision or api.dataset_info(cfg.dataset_id).sha
    save_json(out / "config.json", asdict(cfg))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
    raw = load_dataset(cfg.dataset_id, revision=cfg.dataset_revision)
    labels = raw["train"].features["ner_tags"].feature.names
    ids = {split: list(range(len(ds))) for split, ds in raw.items()}
    if cfg.smoke:
        for split in raw:
            size = 32 if split == "train" else 16
            ids[split] = np.random.default_rng(cfg.seed).choice(len(raw[split]), size, replace=False).tolist()
            raw[split] = raw[split].select(ids[split])
    data, lengths = {}, {}
    for split, ds in raw.items():
        assert not sum(invalid_iob2([labels[t] for t in row["ner_tags"]]) for row in ds)
        data[split] = ds.map(lambda row: align_labels(tokenizer, row["tokens"], row["ner_tags"]),
                             remove_columns=ds.column_names, desc=f"Align {split}")
        lengths[split] = max(map(len, data[split]["input_ids"]))
        if lengths[split] > tokenizer.model_max_length:
            raise ValueError("Sequence exceeds model capacity; implement full-word windows before training")
    sets = {k: {tuple(row["tokens"]) for row in ds} for k, ds in raw.items()}
    overlaps = {f"{a}/{b}": len(sets[a] & sets[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    manifest = {"dataset_id": cfg.dataset_id, "dataset_revision": cfg.dataset_revision,
                "model_id": cfg.model_id, "model_revision": cfg.model_revision, "labels": labels,
                "split_sizes": {k: len(v) for k, v in raw.items()}, "source_indices": ids,
                "max_subtoken_lengths": lengths, "truncated_words": 0,
                "cross_split_exact_sentence_overlaps": overlaps,
                "protocol": "Original official splits retained, including repeated sentences; exact overlaps audited, not removed."}
    save_json(out / "data_manifest.json", manifest)
    # Inspect a real batch with split words, special tokens and padding.
    i = next(i for i, row in enumerate(data["train"]) if len(row["input_ids"]) > len(raw["train"][i]["tokens"]) + 2)
    j = (i + 1) % len(data["train"])
    collator = DataCollatorForTokenClassification(tokenizer)
    batch = collator([data["train"][i], data["train"][j]])
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    sanity = []
    for b, source_id in enumerate([i, j]):
        row = raw["train"][source_id]
        encoding = tokenizer(row["tokens"], is_split_into_words=True)
        wids = encoding.word_ids() + [None] * (batch["input_ids"].shape[1] - len(encoding["input_ids"]))
        for pos, (tok, label, wid) in enumerate(zip(batch["input_ids"][b].tolist(), batch["labels"][b].tolist(), wids)):
            sanity.append({"batch_row": b, "source_row": ids["train"][source_id], "position": pos,
                           "subtoken": tokenizer.convert_ids_to_tokens(tok), "word": row["tokens"][wid] if wid is not None else "",
                           "label_id": label, "label": labels[label] if label != -100 else "IGNORE"})
    pd.DataFrame(sanity).to_csv(out / "alignment_check.csv", index=False)
    print(pd.DataFrame(sanity).to_string(index=False), flush=True)
    print("Data audit:", json.dumps({k: manifest[k] for k in ["split_sizes", "max_subtoken_lengths", "cross_split_exact_sentence_overlaps"]}), flush=True)
    return tokenizer, raw, data, labels, manifest


def loader_for(ds, tokenizer, cfg, train=False):
    return DataLoader(ds, batch_size=cfg.micro_batch, shuffle=train, num_workers=0,
                      generator=torch.Generator().manual_seed(cfg.seed),
                      collate_fn=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8),
                      pin_memory=torch.cuda.is_available())


def metrics_for(references, predictions):
    options = dict(mode="strict", scheme=IOB2, zero_division=0)
    return {"precision": float(precision_score(references, predictions, **options)),
            "recall": float(recall_score(references, predictions, **options)),
            "f1": float(f1_score(references, predictions, **options)),
            "token_accuracy": float(accuracy_score(references, predictions)),
            "conlleval_f1": float(f1_score(references, predictions, zero_division=0)),
            "invalid_predicted_i_transitions": int(sum(invalid_iob2(row) for row in predictions))}


@torch.no_grad()
def evaluate(model, loader, labels, device):
    model.eval()
    total_loss, total_words = 0., 0
    predictions, references = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        count = int((batch["labels"] != -100).sum())
        with amp_context(device):
            output = model(**batch)
        total_loss += output.loss.item() * count
        total_words += count
        pred = output.logits.argmax(-1).cpu().tolist()
        gold = batch["labels"].cpu().tolist()
        for p, g in zip(pred, gold):
            predictions.append([labels[x] for x, y in zip(p, g) if y != -100])
            references.append([labels[y] for y in g if y != -100])
    return {"loss": total_loss / total_words, **metrics_for(references, predictions)}, predictions, references


def freeze_partial(model, top_layers):
    depth = len(model.bert.encoder.layer)
    if not 0 < top_layers < depth:
        raise ValueError("Partial adaptation must train some, but not all, encoder layers")
    model.bert.requires_grad_(False)
    for layer in model.bert.encoder.layer[-top_layers:]:
        layer.requires_grad_(True)


def training_mode(model, partial, top_layers):
    model.train()
    if partial:
        model.bert.embeddings.eval()
        for layer in model.bert.encoder.layer[:-top_layers]:
            layer.eval()


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            digest.update(name.encode())
            digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_method(method, cfg, out, data, tokenizer, labels, device):
    seed_everything(cfg.seed)
    folder = out / method
    folder.mkdir()
    model = AutoModelForTokenClassification.from_pretrained(cfg.model_id, revision=cfg.model_revision,
        num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={s: i for i, s in enumerate(labels)},
        attn_implementation="eager")
    partial = method == "partial_finetuning"
    if partial:
        freeze_partial(model, cfg.top_layers)
    initial_hash = frozen_hash(model) if partial else None
    model.to(device)
    groups = [{"params": list(model.classifier.parameters()), "lr": cfg.head_lr},
              {"params": [p for p in model.bert.parameters() if p.requires_grad], "lr": cfg.backbone_lr}]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == sum(p.numel() for g in groups for p in g["params"])
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    loader = loader_for(data["train"], tokenizer, cfg, True)
    val_loader = loader_for(data["validation"], tokenizer, cfg)
    steps = (len(loader) + cfg.accumulation - 1) // cfg.accumulation * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * .1), steps)
    history, best = [], -1.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, cfg.epochs + 1):
        training_mode(model, partial, cfg.top_layers)
        save_json(out / "progress.json", {"method": method, "epoch": epoch, "batch": 0, "batches": len(loader)})
        synchronize(device)
        start = time.perf_counter()
        total_loss, total_words, batch_index = 0., 0, 0
        iterator = iter(loader)
        while batches := list(itertools.islice(iterator, cfg.accumulation)):
            # Normalize gradients by supervised words in the WHOLE accumulation group.
            group_words = sum(int((b["labels"] != -100).sum()) for b in batches)
            optimizer.zero_grad(set_to_none=True)
            for batch in batches:
                batch = {k: v.to(device) for k, v in batch.items()}
                count = int((batch["labels"] != -100).sum())
                with amp_context(device):
                    output = model(**batch)
                    loss = output.loss * (count / group_words)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                loss.backward()
                total_loss += output.loss.item() * count
                total_words += count
                batch_index += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            if batch_index % 200 == 0:
                save_json(out / "progress.json", {"method": method, "epoch": epoch, "batch": batch_index, "batches": len(loader)})
                print(f"{method}: epoch {epoch}, batch {batch_index}/{len(loader)}", flush=True)
        synchronize(device)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        validation, _, _ = evaluate(model, val_loader, labels, device)
        synchronize(device)
        row = {"epoch": epoch, "train_loss": total_loss / total_words, "train_seconds": train_seconds,
               "validation_seconds": time.perf_counter() - start, **{f"val_{k}": v for k, v in validation.items()}}
        history.append(row)
        save_json(folder / "history.json", history)
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(method, json.dumps(row), flush=True)
        if validation["f1"] > best:
            best, best_epoch = validation["f1"], epoch
            model.save_pretrained(folder / "best_model")
            tokenizer.save_pretrained(folder / "best_model")
    if partial:
        assert frozen_hash(model) == initial_hash, "Frozen parameters changed"
    summary = {"method": method, "best_epoch": best_epoch, "best_val_f1": best, "trainable_parameters": trainable,
               "total_parameters": sum(p.numel() for p in model.parameters()), "frozen_parameters_verified": partial,
               "train_seconds": sum(h["train_seconds"] for h in history),
               "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    save_json(folder / "summary.json", summary)
    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def select_winner(summaries):
    return max(summaries, key=lambda r: (r["best_val_f1"], -r["trainable_parameters"]))["method"]


def entity_set(tags):
    return {(e.tag, e.start, e.end) for e in Entities([tags], IOB2).entities[0]}


def error_analysis(raw, predictions, references, source_ids):
    counts, examples = Counter(), []
    for row, pred, gold, source_id in zip(raw, predictions, references, source_ids):
        expected, found = entity_set(gold), entity_set(pred)
        counts["gold_entities"] += len(expected)
        counts["predicted_entities"] += len(found)
        counts["correct_entities"] += len(expected & found)
        counts["false_positive_entities"] += len(found - expected)
        missing = expected - found
        for tag, start, end in missing:
            if any(a == start and b == end for _, a, b in found):
                kind = "wrong_type_exact_boundary"
            elif any(a < end and start < b for _, a, b in found):
                kind = "overlapping_wrong_boundary"
            else:
                kind = "missed_entity"
            counts[kind] += 1
        if (missing or found - expected) and len(examples) < 20:
            def describe(items):
                return [{"type": t, "start_word": s, "end_word_exclusive": e, "text": " ".join(row["tokens"][s:e])} for t, s, e in sorted(items)]
            examples.append({"source_row": source_id, "text": " ".join(row["tokens"]),
                             "missed_or_wrong": describe(missing), "extra_or_wrong": describe(found - expected)})
    return {"counts": dict(counts), "examples": examples,
            "note": "Gold errors are mutually exclusive (type, boundary, missed). False positives are counted separately. No BIO repair."}


@torch.no_grad()
def predict_words(model, tokenizer, words):
    encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=False)
    if encoded["input_ids"].shape[1] > model.config.max_position_embeddings:
        raise ValueError("Input too long; split at sentence boundaries")
    word_ids = encoded.word_ids()
    model.eval()
    device = next(model.parameters()).device
    predictions = model(**{k: v.to(device) for k, v in encoded.items()}).logits.argmax(-1)[0].tolist()
    tags, seen = [], set()
    for p, w in zip(predictions, word_ids):
        if w is not None and w not in seen:
            tags.append(model.config.id2label[p])
            seen.add(w)
    assert len(tags) == len(words)
    return tags


def run_experiment(cfg, output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    env = {"python": platform.python_version(), "device": str(device), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "precision": "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"}
    save_json(out / "environment.json", env)
    save_json(out / "config.json", asdict(cfg))
    save_json(out / "status.json", {"status": "running", "smoke": cfg.smoke})
    for file in [Path(__file__), Path(__file__).with_name("agnews_experiment.py")]:
        shutil.copy2(file, out / file.name)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name"))
    (out / "requirements-lock.txt").write_text("# CUDA torch: install from cu124 index\n" + "\n".join(packages), encoding="utf-8")
    try:
        tokenizer, raw, data, labels, manifest = prepare_data(cfg, out)
        summaries = [train_method(m, cfg, out, data, tokenizer, labels, device) for m in METHODS]
        winner = select_winner(summaries)
        save_json(out / "selection.json", {"winner": winner, "criterion": "maximum strict IOB2 micro entity F1 on validation; exact tie: fewer trainable parameters",
                  "single_seed": True, "test_used_for_selection": False, "smoke": cfg.smoke})
        for summary in summaries:
            method = summary["method"]
            model = AutoModelForTokenClassification.from_pretrained(out / method / "best_model", attn_implementation="eager").to(device)
            scores, pred, gold = evaluate(model, loader_for(data["test"], tokenizer, cfg), labels, device)
            assert all(len(p) == len(r["tokens"]) for p, r in zip(pred, raw["test"]))
            summary.update({f"test_{k}": v for k, v in scores.items()})
            summary["selected"] = method == winner
            report = classification_report(gold, pred, mode="strict", scheme=IOB2, output_dict=True, zero_division=0)
            save_json(out / method / "test_metrics.json", {**scores, "per_type": report})
            analysis = error_analysis(raw["test"], pred, gold, manifest["source_indices"]["test"])
            save_json(out / method / "error_analysis.json", analysis)
            with (out / method / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
                for i, (p, g) in enumerate(zip(pred, gold)):
                    f.write(json.dumps({"source_row": manifest["source_indices"]["test"][i], "tokens": raw["test"][i]["tokens"],
                                        "reference": g, "prediction": p}, ensure_ascii=False) + "\n")
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        save_json(out / "comparison.json", summaries)
        pd.DataFrame(summaries).to_csv(out / "comparison.csv", index=False)
        make_plot(out)
        if not cfg.smoke:
            delivered = out / "delivered_model_ner"
            shutil.copytree(out / winner / "best_model", delivered)
            write_card(delivered, cfg, summaries, winner)
            model = AutoModelForTokenClassification.from_pretrained(delivered)
            words = ["John", "Smith", "works", "for", "Microsoft", "in", "New", "York", "."]
            save_json(out / "inference_example.json", {"words": words, "tags": predict_words(model, tokenizer, words)})
            del model
            gc.collect()
        write_report(out, cfg, summaries, manifest, winner)
        save_json(out / "status.json", {"status": "complete", "smoke": cfg.smoke})
        print("COMPLETE:", out, "winner:", winner, flush=True)
        return pd.DataFrame(summaries)
    except Exception as exc:
        save_json(out / "status.json", {"status": "failed", "smoke": cfg.smoke, "error": repr(exc)})
        raise


def make_plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, color in zip(METHODS, ["#7451a8", "#098477"]):
        h = pd.read_csv(out / method / "history.csv")
        axes[0].plot(h.epoch, h.train_loss, "o--", color=color, label=method + " train")
        axes[0].plot(h.epoch, h.val_loss, "o-", color=color, label=method + " validation")
        axes[1].plot(h.epoch, h.val_f1, "o-", color=color, label=method)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(h.epoch)
        ax.legend(fontsize=7)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Loss per supervised word")
    axes[1].set_ylabel("Validation strict entity F1 (IOB2)")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)


def write_card(delivered, cfg, summaries, winner):
    selected = next(s for s in summaries if s["selected"])
    text = f"""---
language: en
base_model: {cfg.model_id}
datasets:
- lhoestq/conll2003
pipeline_tag: token-classification
metrics:
- precision
- recall
- f1
---
# BERT-base-cased NER: {winner}

English news entity recognition: PER, ORG, LOC, MISC; IOB2 labels plus O.
Config: `{json.dumps(asdict(cfg))}`.
The best epoch ({selected['best_epoch']}) and method were selected by strict entity F1 on validation.
Test precision {selected['test_precision']:.6f}, recall {selected['test_recall']:.6f}, F1 {selected['test_f1']:.6f}.
Metrics use seqeval 1.2.2, mode=strict, scheme=IOB2; complete entity boundaries and type must match.
Original official train/validation/test splits; exact repeated sentences are retained and audited in data_manifest.json.
First subword of every word is supervised; continuation/special/padding positions use -100.
No sentence truncation. Evaluation uses word-level argmax without repairing invalid BIO transitions.
For matching inference, pass pretokenized words with is_split_into_words=True and take only the first
subtoken prediction for each word (see predict_words in src/ner_experiment.py).

Partial adaptation trains the last two encoder layers and head, keeping embeddings/lower layers frozen in eval mode.
Full adaptation trains all parameters. Both use head LR 1e-3, trainable encoder LR 2e-5, effective batch 16, three epochs.
One seed per configuration: no claim of statistical significance. Not validated for Spanish, other domains,
or entity categories outside CoNLL-2003. Incorrect boundaries, missed entities and wrong types remain possible.

References: https://huggingface.co/datasets/lhoestq/conll2003 ; https://arxiv.org/abs/1810.04805 ;
https://github.com/chakki-works/seqeval . Check upstream dataset/model terms for your intended use.
This local export does not imply Hub publication or collaborator permissions.
"""
    (delivered / "README.md").write_text(text, encoding="utf-8")


def write_report(out, cfg, summaries, manifest, winner):
    lines = ["# NER: resultados medidos", "", "PRUEBA TECNICA: no son resultados finales." if cfg.smoke else "Comparación final; una semilla por configuración.", "",
             f"Particiones oficiales: {manifest['split_sizes']}. Sin truncamiento; máximos de subtokens: {manifest['max_subtoken_lengths']}.",
             f"Frases exactas compartidas entre particiones (conservadas para mantener el benchmark): {manifest['cross_split_exact_sentence_overlaps']}.",
             "Selección: F1 micro de entidades, seqeval strict IOB2. La entidad debe coincidir en tipo, inicio y fin.", "",
             "| Método | Época | F1 validación | Precisión prueba | Recall prueba | F1 prueba | Entrenamiento (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['method']} | {s['best_epoch']} | {s['best_val_f1']:.4f} | {s['test_precision']:.4f} | {s['test_recall']:.4f} | {s['test_f1']:.4f} | {s['train_seconds']:.1f} |")
    lines += ["", f"Ganador por validación: **{winner}**. Prueba se evaluó después de guardar la selección.",
              "Los tiempos excluyen validación y guardado; ambos métodos ejecutan el cuerpo de BERT en cada lote.",
              "La pérdida se promedia por palabras supervisadas, no por frases ni subtokens ignorados.",
              "Una semilla no establece superioridad estadística; diferencias pequeñas pueden depender del azar.",
              "No se ajustaron hiperparámetros con prueba. Las etiquetas O no dominan la métrica principal de entidades.", ""]
    for s in summaries:
        errors = json.loads((out / s["method"] / "error_analysis.json").read_text(encoding="utf-8"))
        lines += [f"## Errores: {s['method']}", "", "```json", json.dumps(errors["counts"], indent=2), "```", "",
                  "Tipos incorrectos, límites incorrectos y entidades omitidas son categorías excluyentes de errores sobre entidades reales. Falsos positivos se cuentan aparte.", ""]
    lines += ["Evidencia: data_manifest.json, alignment_check.csv, historiales, selection.json, predicciones por palabra, métricas por tipo y ejemplos de error.",
              "BERT-base-cased, precisión bf16 cuando la GPU lo admite, semilla 42 y dos grupos de LR (cabeza 1e-3, encoder 2e-5).",
              "Fuentes: https://huggingface.co/datasets/lhoestq/conll2003 ; https://github.com/chakki-works/seqeval ; https://arxiv.org/abs/1810.04805",
              "POS y QA siguen pendientes. El reporte PDF anterior aún no fue actualizado. Modelo exportado localmente, no publicado."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--config")
    args = p.parse_args()
    cfg = Config(**json.loads(Path(args.config).read_text(encoding="utf-8"))) if args.config else Config()
    if args.smoke:
        cfg.smoke, cfg.epochs = True, 1
    run_experiment(cfg, args.output)
