import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pos_experiment as p
from transformers import BertConfig, BertForTokenClassification
class Tests(unittest.TestCase):
 def test_conllu(self):
  x='1-2\tcan\t_\t_\t_\t_\t_\t_\t_\t_\n1\tca\t_\tAUX\t_\t_\t_\t_\t_\t_\n2\tnot\t_\tPART\t_\t_\t_\t_\t_\t_\n2.1\tx\t_\tX\t_\t_\t_\t_\t_\t_\n'
  self.assertEqual(p.parse_conllu(x)[0]['tokens'],['ca','not'])
  with self.assertRaises(ValueError):p.parse_conllu('1\tx\t_\t_')
 def test_metrics(self):
  m=p.metrics_for([['NOUN','VERB']],[['NOUN','NOUN']]);self.assertEqual(m['token_accuracy'],.5)
  self.assertAlmostEqual(m['macro_f1'],(2/3)/17)
 def test_freezing(self):
  m=BertForTokenClassification(BertConfig(hidden_size=24,num_hidden_layers=4,num_attention_heads=4,intermediate_size=48,num_labels=17))
  p.freeze_partial(m,2);p.training_mode(m,True,2)
  self.assertFalse(m.bert.embeddings.training);self.assertFalse(m.bert.encoder.layer[1].training)
  self.assertTrue(m.bert.encoder.layer[2].training)
  self.assertFalse(next(m.bert.encoder.layer[1].parameters()).requires_grad)
  self.assertTrue(next(m.bert.encoder.layer[2].parameters()).requires_grad)
 def test_selection(self):
  self.assertEqual(p.select_winner([{'method':'a','best_val_accuracy':.9,'trainable_parameters':10},{'method':'b','best_val_accuracy':.9,'trainable_parameters':20}]),'a')
if __name__=='__main__':unittest.main()
