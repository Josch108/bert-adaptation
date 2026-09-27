"""Reproducible AG News comparison. Test data never select a model or epoch."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
METHODS = ("feature_based", "full_finetuning")


@dataclass
class Config:
    seed: int = 42
    train_size: int = 12000
    val_size: int = 2000
    test_size: int = 2000
    epochs: int = 3
    micro_batch: int = 8
    accumulation: int = 4
    max_length: int = 128
    head_lr: float = 1e-3
    backbone_lr: float = 2e-5
    weight_decay: float = 0.01
    model_id: str = "google-bert/bert-base-uncased"
    dataset_id: str = "fancyzhx/ag_news"
    model_revision: str | None = None
    dataset_revision: str | None = None
    smoke: bool = False


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_indices(indices, labels, size, seed):
    indices = np.asarray(indices)
    if size > len(indices):
        raise ValueError(f"Requested {size} rows; only {len(indices)} available")
    if size == len(indices):
        return indices.tolist()
    selected, _ = train_test_split(indices, train_size=size, stratify=np.asarray(labels)[indices], random_state=seed)
    return selected.tolist()


def text_key(text):
    return " ".join(text.split()).casefold()


def prepare_data(cfg, out):
    api = HfApi()
    model_revision = cfg.model_revision or api.model_info(cfg.model_id).sha
    dataset_revision = cfg.dataset_revision or api.dataset_info(cfg.dataset_id).sha
    cfg.model_revision, cfg.dataset_revision = model_revision, dataset_revision
    save_json(out / "config.json", asdict(cfg))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=model_revision)
    raw = load_dataset(cfg.dataset_id, revision=dataset_revision)
    # Exclude exact normalized duplicates within/across partitions, retaining first rows.
    test_keys = set()
    unique_test = []
    for i, text in enumerate(raw["test"]["text"]):
        key = text_key(text)
        if key not in test_keys:
            unique_test.append(i)
            test_keys.add(key)
    seen = set(test_keys)
    unique_train = []
    for i, text in enumerate(raw["train"]["text"]):
        key = text_key(text)
        if key not in seen:
            unique_train.append(i)
            seen.add(key)
    pool = select_indices(unique_train, raw["train"]["label"], cfg.train_size + cfg.val_size, cfg.seed)
    train_ids, val_ids = train_test_split(pool, train_size=cfg.train_size, test_size=cfg.val_size,
                                        stratify=np.asarray(raw["train"]["label"])[pool], random_state=cfg.seed)
    test_ids = select_indices(unique_test, raw["test"]["label"], cfg.test_size, cfg.seed)
    assert not set(train_ids) & set(val_ids)
    subsets = {"train": raw["train"].select(train_ids), "validation": raw["train"].select(val_ids),
               "test": raw["test"].select(test_ids)}
    key_sets = {k: {text_key(t) for t in ds["text"]} for k, ds in subsets.items()}
    assert all(not key_sets[a] & key_sets[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
    manifest = {"dataset": cfg.dataset_id, "dataset_revision": dataset_revision,
                "model": cfg.model_id, "model_revision": model_revision,
                "indices": {"train": train_ids, "validation": val_ids, "test": test_ids},
                "source_splits": {"train": "train", "validation": "train", "test": "test"},
                "deduplication": "casefold + whitespace normalization; test precedence; first occurrence",
                "excluded_training_rows": len(raw["train"]) - len(unique_train),
                "excluded_duplicate_test_rows": len(raw["test"]) - len(unique_test),
                "class_counts": {k: np.bincount(ds["label"], minlength=4).tolist() for k, ds in subsets.items()}}
    save_json(out / "splits.json", manifest)
    encoded = {}
    for split, ds in subsets.items():
        ds = ds.map(lambda batch: tokenizer(batch["text"], padding="max_length", truncation=True,
                                            max_length=cfg.max_length), batched=True)
        ds = ds.rename_column("label", "labels")
        encoded[split] = ds.with_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    sample = encoded["train"][0]
    assert sample["input_ids"].shape == (cfg.max_length,)
    assert 0 <= sample["labels"].item() < 4
    print("Split sizes/class counts:", manifest["class_counts"], flush=True)
    return tokenizer, encoded, subsets, manifest


def make_loader(ds, cfg, train=False):
    return DataLoader(ds, batch_size=cfg.micro_batch, shuffle=train, num_workers=0,
                      generator=torch.Generator().manual_seed(cfg.seed), pin_memory=torch.cuda.is_available())


def amp_context(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=device.type == "cuda" and torch.cuda.is_bf16_supported())


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    predictions, references = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp_context(device):
            result = model(**batch)
        total_loss += result.loss.item() * len(batch["labels"])
        predictions.extend(result.logits.argmax(-1).cpu().tolist())
        references.extend(batch["labels"].cpu().tolist())
    return {"loss": total_loss / len(references), "accuracy": float(accuracy_score(references, predictions)),
            "macro_f1": float(f1_score(references, predictions, labels=list(range(4)), average="macro", zero_division=0))}, predictions, references


def backbone_hash(model):
    digest = hashlib.sha256()
    for p in model.bert.parameters():
        digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def select_winner(summaries):
    """No test metrics are consulted; exact ties prefer fewer trained parameters."""
    return max(summaries, key=lambda s: (s["best_val_macro_f1"], -s["trainable_parameters"]))["method"]


def train_method(method, cfg, out, data, tokenizer, revision, device):
    seed_everything(cfg.seed)
    folder = out / method
    folder.mkdir()
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id, revision=revision, num_labels=4, attn_implementation="eager",
        id2label=dict(enumerate(LABELS)), label2id={s: i for i, s in enumerate(LABELS)})
    frozen = method == "feature_based"
    if frozen:
        model.bert.requires_grad_(False)
    frozen_before = backbone_hash(model) if frozen else None
    model.to(device)
    groups = [{"params": list(model.classifier.parameters()), "lr": cfg.head_lr}]
    if not frozen:
        groups.append({"params": list(model.bert.parameters()), "lr": cfg.backbone_lr})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert sum(p.numel() for g in groups for p in g["params"]) == trainable
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    loader = make_loader(data["train"], cfg, train=True)
    val_loader = make_loader(data["validation"], cfg)
    steps_per_epoch = (len(loader) + cfg.accumulation - 1) // cfg.accumulation
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps_per_epoch * cfg.epochs * .1), steps_per_epoch * cfg.epochs)
    history, best_score, best_epoch = [], -1.0, None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, cfg.epochs + 1):
        save_json(out / "progress.json", {"method": method, "epoch": epoch, "batch": 0, "batches": len(loader)})
        model.train()
        if frozen:
            model.bert.eval()  # Freeze dropout as well as weights in the extractor.
            assert not model.bert.training and model.classifier.training
        synchronize(device)
        start = time.perf_counter()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # Weight by examples, including a possibly incomplete final accumulation group.
            group_start = (i // cfg.accumulation) * cfg.accumulation * cfg.micro_batch
            group_examples = min(cfg.micro_batch * cfg.accumulation, len(loader.dataset) - group_start)
            with amp_context(device):
                result = model(**batch)
                loss = result.loss * (len(batch["labels"]) / group_examples)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            total_loss += result.loss.item() * len(batch["labels"])
            if (i + 1) % cfg.accumulation == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (i + 1) % 200 == 0:
                save_json(out / "progress.json", {"method": method, "epoch": epoch, "batch": i + 1, "batches": len(loader)})
                print(f"{method} epoch {epoch}: {i+1}/{len(loader)} batches", flush=True)
        synchronize(device)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        validation, _, _ = evaluate(model, val_loader, device)
        synchronize(device)
        row = {"epoch": epoch, "train_loss": total_loss / len(loader.dataset),
               "train_seconds": train_seconds, "validation_seconds": time.perf_counter() - start,
               **{f"val_{k}": v for k, v in validation.items()}}
        history.append(row)
        save_json(folder / "history.json", history)
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(method, json.dumps(row), flush=True)
        # Exact ties retain the earlier epoch.
        if validation["macro_f1"] > best_score:
            best_score, best_epoch = validation["macro_f1"], epoch
            model.save_pretrained(folder / "best_model")
            tokenizer.save_pretrained(folder / "best_model")
    if frozen:
        assert backbone_hash(model) == frozen_before, "Frozen BERT weights changed"
    summary = {"method": method, "trainable_parameters": trainable,
               "total_parameters": sum(p.numel() for p in model.parameters()),
               "best_epoch": best_epoch, "best_val_macro_f1": best_score,
               "train_seconds": sum(r["train_seconds"] for r in history),
               "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
               "frozen_backbone_verified": frozen}
    save_json(folder / "summary.json", summary)
    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def run_experiment(cfg, output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an experiment.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    environment = {"python": platform.python_version(), "platform": platform.platform(), "torch_cuda": torch.version.cuda,
                   "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "precision": "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32",
                   "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "datasets", "numpy", "scikit-learn", "huggingface-hub"]}}
    save_json(out / "config.json", asdict(cfg))
    save_json(out / "environment.json", environment)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name"))
    (out / "requirements-lock.txt").write_text("# Environment snapshot; CUDA torch requires the cu124 wheel index.\n" + "\n".join(packages) + "\n", encoding="utf-8")
    save_json(out / "status.json", {"status": "running", "smoke": cfg.smoke})
    shutil.copy2(__file__, out / "agnews_experiment.py")
    print(json.dumps(environment), flush=True)
    try:
        tokenizer, data, raw, manifest = prepare_data(cfg, out)
        summaries = [train_method(m, cfg, out, data, tokenizer, manifest["model_revision"], device) for m in METHODS]
        # Validation decides before ANY test prediction. Exact ties favor fewer trainable parameters.
        winner = select_winner(summaries)
        gap = abs(summaries[0]["best_val_macro_f1"] - summaries[1]["best_val_macro_f1"])
        selection = {"winner": winner, "criterion": "highest validation macro-F1; exact tie: fewer trainable parameters",
                     "validation_gap": gap, "single_seed_caution": "A single seed does not establish statistical superiority.",
                     "smoke": cfg.smoke}
        save_json(out / "selection.json", selection)
        for summary in summaries:
            method = summary["method"]
            model = AutoModelForSequenceClassification.from_pretrained(out / method / "best_model", attn_implementation="eager").to(device)
            scores, preds, refs = evaluate(model, make_loader(data["test"], cfg), device)
            summary.update({f"test_{k}": v for k, v in scores.items()})
            summary["selected"] = method == winner
            save_json(out / method / "test_metrics.json", {**scores, "confusion_matrix": confusion_matrix(refs, preds, labels=list(range(4))).tolist(),
                        "classification_report": classification_report(refs, preds, labels=list(range(4)), target_names=LABELS, output_dict=True, zero_division=0)})
            with (out / method / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
                for i, (pred, ref) in enumerate(zip(preds, refs)):
                    f.write(json.dumps({"source_row": manifest["indices"]["test"][i], "text": raw["test"][i]["text"],
                                        "reference": ref, "prediction": pred}, ensure_ascii=False) + "\n")
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        save_json(out / "comparison.json", summaries)
        pd.DataFrame(summaries).to_csv(out / "comparison.csv", index=False)
        make_plot(out)
        if not cfg.smoke:
            delivered = out / "delivered_model_agnews"
            shutil.copytree(out / winner / "best_model", delivered)
            selected = next(s for s in summaries if s["selected"])
            card = f"""---
language: en
base_model: {cfg.model_id}
datasets:
- fancyzhx/ag_news
pipeline_tag: text-classification
metrics:
- accuracy
- f1
---
# AG News: BERT-base, {winner}

## Training and selection
English news classification: {LABELS}. Same standard BERT pooled-output linear head in both alternatives.
Configuration: `{json.dumps(asdict(cfg))}`.
Best checkpoint: epoch {selected['best_epoch']}, chosen by validation macro-F1.
Validation macro-F1: {selected['best_val_macro_f1']:.6f}.
Held-out test accuracy: {selected['test_accuracy']:.6f}; macro-F1: {selected['test_macro_f1']:.6f}.
Compared frozen BERT + linear head with full fine-tuning. Both used the same split and seed.
The full fine-tuning optimizer used separate head/backbone learning rates.
Only one seed was run; these results do not establish statistical superiority.

## Data and limitations
Stratified subsets: {cfg.train_size} train, {cfg.val_size} validation, {cfg.test_size} test.
Train/validation come from the official training split; test comes from the official test split.
Exact normalized duplicate text was excluded across partitions. See splits.json for source indices and revisions.
Input is truncated to {cfg.max_length} tokens. Not validated for Spanish, other domains, or other categories.
Automated predictions can be wrong; assess errors for the intended application.

## Usage
```python
from transformers import pipeline
classifier = pipeline('text-classification', model='PATH_OR_HUB_REPO')
print(classifier('A new satellite was launched today.'))
```

## References
- https://huggingface.co/{cfg.model_id}
- https://huggingface.co/datasets/{cfg.dataset_id}
- https://arxiv.org/abs/1810.04805

Publication/access permissions are not implied by this local export.
"""
            (delivered / "README.md").write_text(card, encoding="utf-8")
        write_report(out, cfg, summaries, selection, environment)
        save_json(out / "status.json", {"status": "complete", "smoke": cfg.smoke})
        print("COMPLETE", out, "winner:", winner, flush=True)
        return pd.DataFrame(summaries)
    except Exception as exc:
        save_json(out / "status.json", {"status": "failed", "error": repr(exc), "smoke": cfg.smoke})
        raise


def make_plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in METHODS:
        h = pd.read_csv(out / method / "history.csv")
        axes[0].plot(h.epoch, h.train_loss, "o--", label=f"{method}: train")
        axes[0].plot(h.epoch, h.val_loss, "o-", label=f"{method}: validation")
        axes[1].plot(h.epoch, h.val_macro_f1, "o-", label=method)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_ylabel("Validation macro-F1")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)


def write_report(out, cfg, summaries, selection, env):
    lines = ["# AG News: resultados medidos", "",
             "PRUEBA TECNICA: no usar estas cifras como resultados finales." if cfg.smoke else "Comparacion final de dos configuraciones; una semilla por configuracion.", "",
             f"Equipo: {env['gpu'] or env['device']}; precision: {env['precision']}.",
             f"Datos: {cfg.train_size} entrenamiento / {cfg.val_size} validacion / {cfg.test_size} prueba; semilla {cfg.seed}.",
             "Misma arquitectura BERT + pooler + clasificador lineal para aislar el efecto de congelar o ajustar el cuerpo.",
             "Los tiempos incluyen el forward de BERT en cada epoca; no hay cache de embeddings.", "",
             "| Metodo | Epoca elegida | F1 validacion | Accuracy prueba | F1 prueba | Entrenamiento (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['method']} | {s['best_epoch']} | {s['best_val_macro_f1']:.4f} | {s['test_accuracy']:.4f} | {s['test_macro_f1']:.4f} | {s['train_seconds']:.1f} |")
    lines.extend(["", f"Seleccion por validacion: **{selection['winner']}**. Diferencia de F1: {selection['validation_gap']:.4f}.",
                  "Una semilla no permite afirmar superioridad estadistica; diferencias pequenas pueden depender del azar.",
                  "Se evaluaron los mejores checkpoints de ambas alternativas en prueba despues de fijar la seleccion.",
                  "No se afirma una causa linguistica de las diferencias sin analizar ejemplos y realizar experimentos adicionales.", "",
                  "Evidencia: config.json, environment.json, splits.json, selection.json, comparison.csv, historiales por epoca y test_predictions.jsonl.",
                  "Los resultados de NER, POS y QA del reporte anterior siguen pendientes de verificacion."])
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", help="Reuse a saved config.json, including pinned dataset/model revisions")
    args = parser.parse_args()
    cfg = Config(**json.loads(Path(args.config).read_text(encoding="utf-8"))) if args.config else Config()
    if args.smoke:
        cfg = Config(train_size=32, val_size=16, test_size=16, epochs=1, smoke=True)
    run_experiment(cfg, args.output)
