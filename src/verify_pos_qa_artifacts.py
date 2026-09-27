"""Independent offline checks of new POS/QA evidence; does not train models."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import re
import string
import sys

root = Path(sys.argv[1]).resolve()
read = lambda p: json.loads(p.read_text(encoding="utf-8"))

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def close(actual, expected):
    assert math.isclose(actual, expected, abs_tol=1e-10), (actual, expected)

def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())

summary = {}
for task in ("pos", "qa"):
    run = root / "runs" / f"{task}_verified"
    status = read(run / "status.json")
    assert status["status"] in {"complete", "completed"} and not status["smoke"]
    rows = read(run / "comparison.json")
    selection = read(run / "selection.json")
    criterion = "best_val_accuracy" if task == "pos" else "best_val_f1"
    expected = max(rows, key=lambda r: (r[criterion], -r["trainable_parameters"]))["method"]
    assert selection["winner"] == expected
    checks = []
    test_identity = None
    for row in rows:
        folder = run / row["method"]
        history = read(folder / "history.json")
        validation_metric = "val_token_accuracy" if task == "pos" else "val_f1"
        best_epoch = max(history, key=lambda epoch: epoch[validation_metric])
        assert row["best_epoch"] == best_epoch["epoch"]
        close(row[criterion], best_epoch[validation_metric])
        close(row["train_seconds"], sum(epoch["train_seconds"] for epoch in history))
        if task == "pos":
            predictions = [json.loads(line) for line in (folder / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            identity = [(p["source_row"], p["tokens"], p["reference"]) for p in predictions]
            pairs = []
            for p in predictions:
                assert len(p["tokens"]) == len(p["reference"]) == len(p["prediction"])
                pairs.extend(zip(p["reference"], p["prediction"]))
            labels = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
            assert all(g in labels and p in labels for g, p in pairs)
            correct = sum(g == p for g, p in pairs)
            accuracy = correct / len(pairs)
            f1 = []
            for label in labels:
                tp = sum(g == p == label for g, p in pairs)
                denom = sum(g == label for g, p in pairs) + sum(p == label for g, p in pairs)
                f1.append(2 * tp / denom if denom else 0)
            macro = sum(f1) / len(labels)
            close(accuracy, row["test_token_accuracy"])
            close(macro, row["test_macro_f1"])
            checks.append({"method": row["method"], "sentences": len(predictions), "words": len(pairs), "correct": correct, "accuracy": accuracy, "macro_f1": macro})
        else:
            refs = read(folder / "test_references.json")
            preds = read(folder / "test_predictions.json")
            assert len({r["id"] for r in refs}) == len(refs) == len(preds)
            assert set(preds) == {r["id"] for r in refs}
            identity = refs
            em_total = f1_total = 0
            for ref in refs:
                answer = preds[ref["id"]]
                assert answer in ref["context"]
                pred = normalize(answer)
                em = f1 = 0
                for gold in ref["answers"]["text"]:
                    gold = normalize(gold)
                    em = max(em, int(pred == gold))
                    pt, gt = pred.split(), gold.split()
                    overlap = sum((Counter(pt) & Counter(gt)).values())
                    if overlap:
                        precision, recall = overlap / len(pt), overlap / len(gt)
                        f1 = max(f1, 2 * precision * recall / (precision + recall))
                em_total += em
                f1_total += f1
            em, f1 = 100 * em_total / len(refs), 100 * f1_total / len(refs)
            close(em, row["test_exact_match"])
            close(f1, row["test_f1"])
            checks.append({"method": row["method"], "questions": len(refs), "exact_match": em, "f1": f1, "all_answers_from_context": True})
        if test_identity is None:
            test_identity = identity
        else:
            assert test_identity == identity, "Alternatives evaluated on different examples"
    export = run / f"delivered_model_{task}"
    original = run / expected / "best_model"
    model_hash = digest(export / "model.safetensors")
    assert model_hash == digest(original / "model.safetensors")
    assert read(export / "config.json") == read(original / "config.json")
    source = root / "src" / f"{task}_experiment.py"
    snapshot = run / f"{task}_experiment.py"
    source_identical = digest(source) == digest(snapshot)
    recovery_verified = False
    if not source_identical:
        assert task == "qa" and (run / "utf8_recovery.json").exists()
        recovery = read(run / "utf8_recovery.json")
        assert digest(snapshot) == recovery["training_source_sha256"]
        assert digest(source) == recovery["corrected_source_sha256"]
        expected_source = snapshot.read_text(encoding="utf-8").replace(
            "(out/winner/'test_predictions.json').read_text()",
            "(out/winner/'test_predictions.json').read_text(encoding='utf-8')")
        assert source.read_text(encoding="utf-8") == expected_source
        assert recovery["predictions_identical"] and not recovery["training_repeated"]
        assert read(run / "reload_verification.json")["all_test_predictions_identical"]
        recovery_verified = True
    notebook_name = "03_pos_tagging_ud_ewt.ipynb" if task == "pos" else "04_extractive_qa_squad.ipynb"
    nb = read(root / "notebooks" / notebook_name)
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(c["execution_count"] is not None for c in code)
    assert not any(o["output_type"] == "error" for c in code for o in c.get("outputs", []))
    summary[task] = {"status": "verified", "selected_by_validation": expected, "same_test_examples": True, "source_matches_executed_snapshot": source_identical, "only_documented_utf8_read_fix": recovery_verified, "export_matches_selected_checkpoint": True, "model_sha256": model_hash, "executed_code_cells": len(code), "recomputed_metrics": checks}

destination = root / "report" / "POS_QA_VERIFICATION.json"
destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
