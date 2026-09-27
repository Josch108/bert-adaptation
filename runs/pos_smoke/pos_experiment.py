"""UD English EWT: frozen versus partial BERT, selected by word accuracy."""
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
from datasets import Dataset, DatasetDict
import requests
from sklearn.metrics import accuracy_score as word_accuracy, f1_score as word_f1, classification_report as word_report, confusion_matrix
from huggingface_hub import HfApi
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import Entities, IOB2
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, get_linear_schedule_with_warmup

METHODS = ("frozen", "partial_finetuning")
UPOS = "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split()


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
    model_id: str = "google-bert/bert-base-uncased"
    dataset_id: str = "UniversalDependencies/UD_English-EWT"
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
    cfg.dataset_revision = cfg.dataset_revision or "r2.15"
    save_json(out / "config.json", asdict(cfg))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
    raw, sources = {}, {}
    for split, suffix in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        url = f"https://raw.githubusercontent.com/{cfg.dataset_id}/{cfg.dataset_revision}/en_ewt-ud-{suffix}.conllu"
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        content = response.content
        (out / f"{split}.conllu").write_bytes(content)
        rows = parse_conllu(content.decode("utf-8"))
        raw[split] = Dataset.from_list(rows)
        sources[split] = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
    raw = DatasetDict(raw)
    save_json(out / "source_files.json", sources)
    labels = UPOS
    ids = {split: list(range(len(ds))) for split, ds in raw.items()}
    if cfg.smoke:
        for split in raw:
            size = 32 if split == "train" else 16
            ids[split] = np.random.default_rng(cfg.seed).choice(len(raw[split]), size, replace=False).tolist()
            raw[split] = raw[split].select(ids[split])
    data, lengths = {}, {}
    for split, ds in raw.items():
        data[split] = ds.map(lambda row: align_labels(tokenizer, row["tokens"], row["upos"]),
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


def parse_conllu(text):
    rows, words, tags = [], [], []
    for line in text.splitlines() + [""]:
        if not line.strip():
            if words:
                rows.append({"tokens": words, "upos": tags})
            words, tags = [], []
        elif not line.startswith("#"):
            cols = line.split("\t")
            if not cols[0].isdigit():
                continue  # Skip multiword surface rows and empty syntactic nodes.
            if cols[3] not in UPOS:
                raise ValueError(f"Missing or unsupported UPOS: {cols[3]}")
            words.append(cols[1]); tags.append(UPOS.index(cols[3]))
    return rows


def metrics_for(references, predictions):
    gold = list(itertools.chain.from_iterable(references))
    pred = list(itertools.chain.from_iterable(predictions))
    return {"token_accuracy": float(word_accuracy(gold, pred)),
            "macro_f1": float(word_f1(gold, pred, labels=UPOS, average="macro", zero_division=0))}


def per_type_report(references, predictions):
    return word_report(list(itertools.chain.from_iterable(references)),
                       list(itertools.chain.from_iterable(predictions)), labels=UPOS, output_dict=True, zero_division=0)


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
    partial = True
    if method == "partial_finetuning":
        freeze_partial(model, cfg.top_layers)
    else:
        model.bert.requires_grad_(False)
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
        if method == "frozen":
            model.bert.eval()
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
        if validation["token_accuracy"] > best:
            best, best_epoch = validation["token_accuracy"], epoch
            model.save_pretrained(folder / "best_model")
            tokenizer.save_pretrained(folder / "best_model")
    if partial:
        assert frozen_hash(model) == initial_hash, "Frozen parameters changed"
    summary = {"method": method, "best_epoch": best_epoch, "best_val_accuracy": best, "trainable_parameters": trainable,
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
    return max(summaries, key=lambda r: (r["best_val_accuracy"], -r["trainable_parameters"]))["method"]


def error_analysis(raw, predictions, references, source_ids):
    confusion, examples = Counter(), []
    total = errors = 0
    for row, pred, gold, idx in zip(raw, predictions, references, source_ids):
        for word, p, g in zip(row["tokens"], pred, gold):
            total += 1
            if p != g:
                errors += 1
                confusion[(g, p)] += 1
                if len(examples) < 30:
                    examples.append({"source_row": idx, "word": word, "gold": g, "prediction": p, "sentence": " ".join(row["tokens"])})
    return {"counts": {"words": total, "errors": errors, "correct": total-errors},
            "top_confusions": [{"gold": g, "prediction": p, "count": n} for (g,p),n in confusion.most_common(20)], "examples": examples}


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
        save_json(out / "selection.json", {"winner": winner, "criterion": "maximum word accuracy on validation; exact tie: fewer trainable parameters",
                  "single_seed": True, "test_used_for_selection": False, "smoke": cfg.smoke})
        for summary in summaries:
            method = summary["method"]
            model = AutoModelForTokenClassification.from_pretrained(out / method / "best_model", attn_implementation="eager").to(device)
            scores, pred, gold = evaluate(model, loader_for(data["test"], tokenizer, cfg), labels, device)
            assert all(len(p) == len(r["tokens"]) for p, r in zip(pred, raw["test"]))
            summary.update({f"test_{k}": v for k, v in scores.items()})
            summary["selected"] = method == winner
            report = per_type_report(gold, pred)
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
            delivered = out / "delivered_model_pos"
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
        axes[1].plot(h.epoch, h.val_token_accuracy, "o-", color=color, label=method)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(h.epoch)
        ax.legend(fontsize=7)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Loss per supervised word")
    axes[1].set_ylabel("Validation word accuracy")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)


def write_card(delivered, cfg, summaries, winner):
    s = next(s for s in summaries if s["selected"])
    text = f"""---
language: en
base_model: {cfg.model_id}
pipeline_tag: token-classification
metrics:
- accuracy
- f1
---
# BERT English UPOS ({winner})
17 Universal POS tags, UD English EWT r2.15 official train/dev/test.
Source: https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.15
Configuration: {json.dumps(asdict(cfg))}
Best epoch: {s['best_epoch']}, selected by validation word accuracy.
Test accuracy: {s['test_token_accuracy']:.6f}; macro-F1: {s['test_macro_f1']:.6f}.
Only first subtokens are supervised; special/continuation/padding labels are -100.
No truncation. CoNLL-U multiword surface rows and empty nodes are excluded, retaining syntactic words.
Use pretokenized words and first-subtoken argmax (predict_words). Labels are grammatical categories, not entities.
One seed; no statistical significance claim. Not validated for Spanish or other domains.
Frozen BERT stays in eval mode. Partial trains only top two layers and classifier.
Three epochs, head LR 1e-3, encoder LR 2e-5, effective batch 16.
Local export only. Review upstream model and dataset licenses before redistribution.
"""
    (delivered / "README.md").write_text(text, encoding="utf-8")


def write_report(out, cfg, summaries, manifest, winner):
    lines = ["# POS: resultados medidos", "", "Prueba técnica; no final." if cfg.smoke else "Una semilla (42), tres épocas por configuración.",
             f"Particiones oficiales UD English EWT r2.15: {manifest['split_sizes']}.",
             f"Máximos de subtokens: {manifest['max_subtoken_lengths']}. Sin truncamiento.",
             f"Frases repetidas entre particiones, conservadas: {manifest['cross_split_exact_sentence_overlaps']}.",
             "Selección por accuracy de palabras en validación; empate entre métodos: menos parámetros entrenables.",
             "Prueba se evalúa después de guardar la selección. Macro-F1 incluye las 17 etiquetas UPOS.", "",
             "| Método | Época | Accuracy validación | Accuracy prueba | Macro-F1 prueba | Entrenamiento s |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['method']} | {s['best_epoch']} | {s['best_val_accuracy']:.4f} | {s['test_token_accuracy']:.4f} | {s['test_macro_f1']:.4f} | {s['train_seconds']:.1f} |")
    lines += ["", f"Seleccionado: **{winner}**.", "Tiempos excluyen validación y guardado; no son latencia de inferencia.",
              "La pérdida se pondera por palabras supervisadas. Solo el primer subtoken recibe etiqueta.",
              "Las filas multipalabra y nodos vacíos de CoNLL-U no son palabras sintácticas y se omiten; etiquetas desconocidas abortan la carga.",
              "Modelo base uncased: puede limitar distinciones apoyadas en mayúsculas como PROPN/NOUN.",
              "Una semilla no demuestra superioridad estadística. Sin búsqueda de hiperparámetros usando prueba."]
    for s in summaries:
        e = json.loads((out / s['method'] / 'error_analysis.json').read_text())
        lines += ["", f"## Errores {s['method']}", json.dumps(e['counts']), "```json", json.dumps(e['top_confusions'][:10], indent=2), "```"]
    lines += ["", "Fuentes: https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.15 ; https://universaldependencies.org/format.html",
              "Procedencia y SHA256: source_files.json. Entorno bloqueado, predicciones, historias y exportación dentro del directorio de ejecución."]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


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
