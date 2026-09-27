"""Resume ONLY final UTF-8 reload verification after completed training/evaluation."""
from pathlib import Path
import json, hashlib, shutil, difflib
from datasets import Dataset
from qa_experiment import Config, AutoTokenizer, AutoModelForQuestionAnswering, build_features, evaluate, torch, save_json
out=Path(__file__).resolve().parent/'runs'/'qa_verified'
status=json.loads((out/'status.json').read_text(encoding='utf-8'))
assert status['status']=='failed'
shutil.copy2(out/'status.json',out/'status_before_utf8_recovery.json')
cfg=Config(**json.loads((out/'config.json').read_text(encoding='utf-8')))
winner=json.loads((out/'selection.json').read_text(encoding='utf-8'))['winner']
raw=Dataset.from_list(json.loads((out/winner/'test_references.json').read_text(encoding='utf-8')))
tok=AutoTokenizer.from_pretrained(out/'delivered_model_qa')
features=raw.map(lambda b:build_features(b,tok,cfg),batched=True,remove_columns=raw.column_names)
torch.set_num_threads(8)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=AutoModelForQuestionAnswering.from_pretrained(out/'delivered_model_qa',attn_implementation='sdpa').to(device)
scores,preds=evaluate(model,raw,features,cfg,device)
expected=json.loads((out/winner/'test_predictions.json').read_text(encoding='utf-8'))
assert preds==expected
save_json(out/'reload_verification.json',{'all_test_predictions_identical':True,'scores':scores,'verification_recovered_with_explicit_utf8':True})
source=Path(__file__).with_name('qa_experiment.py')
original=source.read_text(encoding='utf-8')
corrected=original.replace("(out/winner/'test_predictions.json').read_text()", "(out/winner/'test_predictions.json').read_text(encoding='utf-8')")
assert corrected!=original
(out/'utf8_fix.patch').write_text(''.join(difflib.unified_diff(original.splitlines(keepends=True),corrected.splitlines(keepends=True),fromfile='executed/qa_experiment.py',tofile='src/qa_experiment.py')),encoding='utf-8')
source.write_text(corrected,encoding='utf-8')
shutil.copy2(__file__,out/'recover_utf8_verification.py')
save_json(out/'utf8_recovery.json',{'reason':'Windows cp1252 default cannot correctly decode UTF-8 predictions; training/evaluation artifacts unchanged','training_source_snapshot':'qa_experiment.py','training_source_sha256':hashlib.sha256((out/'qa_experiment.py').read_bytes()).hexdigest(),'corrected_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'only_source_change':"test_predictions.json read_text() -> read_text(encoding='utf-8')",'predictions_identical':True,'training_repeated':False})
save_json(out/'status.json',{'status':'completed','smoke':False,'final_verification_recovered_utf8':True})
print(scores,flush=True)
