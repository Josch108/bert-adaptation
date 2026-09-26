"""Offline correctness checks; tiny random BERT is not a performance benchmark."""
import gc
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from datasets import Dataset
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast, DataCollatorForTokenClassification
from ner_experiment import Config, align_labels, entity_set, freeze_partial, metrics_for, per_type_report, select_winner, train_method, training_mode


class NERTests(unittest.TestCase):
    def test_strict_entities_require_complete_boundary_type_and_bio(self):
        gold = [["B-LOC", "I-LOC", "O"]]
        self.assertEqual(metrics_for(gold, gold)["f1"], 1)
        self.assertEqual(metrics_for(gold, [["B-LOC", "O", "O"]])["f1"], 0)
        self.assertEqual(metrics_for(gold, [["B-ORG", "I-ORG", "O"]])["f1"], 0)
        malformed = metrics_for(gold, [["I-LOC", "I-LOC", "O"]])
        self.assertEqual(malformed["f1"], 0)
        self.assertEqual(malformed["conlleval_f1"], 1)
        self.assertEqual(entity_set(gold[0]), {("LOC", 0, 2)})
        self.assertEqual(json.loads(json.dumps(per_type_report(gold, gold)))["LOC"]["support"], 1)

    def test_selection_uses_validation_not_test(self):
        rows = [dict(method="partial", best_val_f1=.9, trainable_parameters=10, test_f1=.1),
                dict(method="full", best_val_f1=.8, trainable_parameters=100, test_f1=.99)]
        self.assertEqual(select_winner(rows), "partial")
        rows[1]["best_val_f1"] = .9
        self.assertEqual(select_winner(rows), "partial")

    def test_alignment_freezing_training_and_checkpoint(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            (base / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nJohn\nplay\n##ing\n", encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(base / "vocab.txt"), do_lower_case=False)
            tokenizer.save_pretrained(base)
            labels = ["O", "B-PER", "I-PER"]
            row = align_labels(tokenizer, ["John", "playing"], [1, 0])
            self.assertEqual(row["labels"], [-100, 1, 0, -100, -100])
            short = align_labels(tokenizer, ["John"], [1])
            batch = DataCollatorForTokenClassification(tokenizer)([row, short])
            self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
            torch.manual_seed(42)
            model = BertForTokenClassification(BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=3,
                num_attention_heads=2, intermediate_size=32, max_position_embeddings=32, num_labels=3,
                id2label=dict(enumerate(labels)), label2id={s: i for i, s in enumerate(labels)}))
            original = {k: v.detach().clone() for k, v in model.named_parameters()}
            model.save_pretrained(base)
            freeze_partial(model, 1)
            training_mode(model, True, 1)
            self.assertFalse(model.bert.embeddings.training)
            self.assertFalse(model.bert.encoder.layer[0].training)
            self.assertTrue(model.bert.encoder.layer[-1].training)
            ds = Dataset.from_list([row, short, row, short, row])
            cfg = Config(model_id=str(base), top_layers=1, epochs=1, micro_batch=2, accumulation=2)
            for method in ["partial_finetuning", "full_finetuning"]:
                train_method(method, cfg, root, {"train": ds, "validation": ds}, tokenizer, labels, torch.device("cpu"))
                loaded = BertForTokenClassification.from_pretrained(root / method / "best_model")
                lower = "bert.embeddings.word_embeddings.weight"
                params = dict(loaded.named_parameters())
                self.assertEqual(torch.equal(params[lower], original[lower]), method == "partial_finetuning")
                upper = "bert.encoder.layer.2.output.dense.weight"
                self.assertFalse(torch.equal(params[upper], original[upper]))
                self.assertFalse(torch.equal(params["classifier.weight"], original["classifier.weight"]))
                self.assertTrue(torch.isfinite(loaded(**batch).loss))
                del loaded, params
                gc.collect()


if __name__ == "__main__":
    unittest.main()
