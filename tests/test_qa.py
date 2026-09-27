import unittest, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qa_experiment import *
class QA(unittest.TestCase):
 def test_metrics(self):
  self.assertEqual(answer_score('The, CAT!','cat'),(1,1.))
  self.assertAlmostEqual(answer_score('red car','car')[1],2/3)
  raw=[{'id':'x','answers':{'text':['red car','car']}}]
  self.assertEqual(metrics(raw,{'x':'car'})['f1'],100)
 def test_decode_masks_and_length(self):
  raw=[{'id':'x','context':'red car'}]
  f=[{'example_id':'x','offset_mapping':[None,[0,3],[4,7],None]}]
  pred=postprocess(raw,f,np.array([[99,1,5,100]]),np.array([[99,1,5,100]]))
  self.assertEqual(pred['x'],'car')
  pred=postprocess(raw,f,np.array([[0,10,1,0]]),np.array([[0,1,10,0]]),max_answer_length=1)
  self.assertNotEqual(pred['x'],'red car')
 def test_overflow_alignment(self):
  tok=AutoTokenizer.from_pretrained('google-bert/bert-base-uncased',local_files_only=True)
  context=' '.join(['hello']*500)+' London '+ ' '.join(['world']*100)
  cfg=Config(max_length=128,stride=32)
  b={'id':['x'],'question':['Which city?'],'context':[context],'answers':[{'answer_start':[context.index('London')],'text':['London']}]}
  f=build_features(b,tok,cfg)
  self.assertGreater(len(f['input_ids']),1);self.assertIn('token_type_ids',f)
  positives=0;negatives=0
  for i,s in enumerate(f['start_positions']):
   e=f['end_positions'][i]
   if s==0:negatives+=1;self.assertEqual(e,0)
   else:
    positives+=1;self.assertEqual(context[f['offset_mapping'][i][s][0]:f['offset_mapping'][i][e][1]],'London')
    self.assertEqual(f['token_type_ids'][i][s],1)
   self.assertIsNone(f['offset_mapping'][i][0])
  self.assertGreater(positives,0);self.assertGreater(negatives,0)
 def test_partial_answer_is_negative(self):
  tok=AutoTokenizer.from_pretrained('google-bert/bert-base-uncased',local_files_only=True)
  context=' '.join(['hello']*150)
  b={'id':['x'],'question':['What?'],'context':[context],'answers':[{'answer_start':[0],'text':[context]}]}
  f=build_features(b,tok,Config(max_length=64,stride=16))
  self.assertTrue(all(s==0 and e==0 for s,e in zip(f['start_positions'],f['end_positions'])))
 def test_best_span_across_windows(self):
  raw=[{'id':'x','context':'red car blue bus'}]
  f=[{'example_id':'x','offset_mapping':[None,[0,3],[4,7],None]},{'example_id':'x','offset_mapping':[None,[8,12],[13,16],None]}]
  pred=postprocess(raw,f,np.array([[100,1,1,100],[100,8,1,100]]),np.array([[100,1,1,100],[100,1,8,100]]))
  self.assertEqual(pred['x'],'blue bus')
if __name__=='__main__' :unittest.main(verbosity=2)
