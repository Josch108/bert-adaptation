"""Small, offline checks for freezing, checkpoint recovery, and selection isolation."""
import json
import gc
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from datasets import Dataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from agnews_experiment import Config, backbone_hash, select_winner, train_method


class ExperimentTests(unittest.TestCase):
    def test_selection_ignores_test_and_handles_ties(self):
        rows = [dict(method="frozen", best_val_macro_f1=.8, trainable_parameters=100, test_macro_f1=.1),
                dict(method="full", best_val_macro_f1=.7, trainable_parameters=10000, test_macro_f1=.99)]
        self.assertEqual(select_winner(rows), "frozen")
        rows[1]["best_val_macro_f1"] = .8
        self.assertEqual(select_winner(rows), "frozen")
        rows[1]["best_val_macro_f1"] = .9
        self.assertEqual(select_winner(rows), "full")

    def test_real_training_freezes_only_feature_backbone_and_reloads(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            (base / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nnews\nteam\nmarket\n", encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(base / "vocab.txt"))
            tokenizer.save_pretrained(base)
            torch.manual_seed(42)
            model = BertForSequenceClassification(BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, max_position_embeddings=16, num_labels=4))
            original_hash = backbone_hash(model)
            original_head = model.classifier.weight.detach().clone()
            model.save_pretrained(base)
            encoded = tokenizer(["news team", "market", "team news", "news", "market news"], padding="max_length", max_length=8)
            encoded["labels"] = [0, 1, 2, 3, 0]
            ds = Dataset.from_dict(encoded).with_format("torch")
            cfg = Config(model_id=str(base), epochs=1, micro_batch=2, accumulation=2, max_length=8)
            for method in ["feature_based", "full_finetuning"]:
                summary = train_method(method, cfg, root, {"train": ds, "validation": ds}, tokenizer, None, torch.device("cpu"))
                restored = BertForSequenceClassification.from_pretrained(root / method / "best_model")
                self.assertEqual(summary["best_epoch"], 1)
                self.assertTrue(torch.isfinite(restored(**{k: v.unsqueeze(0) for k, v in ds[0].items()}).loss))
                self.assertFalse(torch.equal(original_head, restored.classifier.weight))
                self.assertEqual(backbone_hash(restored) == original_hash, method == "feature_based")
                self.assertEqual(len(json.loads((root / method / "history.json").read_text())), 1)
                del restored
                gc.collect()


if __name__ == "__main__":
    unittest.main()
