from pathlib import Path
import json
from datasets import Dataset
from qa_experiment import *
out=Path(__file__).resolve().parent/'runs'/'qa_verified'
cfg=Config(**json.loads((out/'config.json').read_text(encoding='utf-8')))
torch.set_num_threads(8)
tok=AutoTokenizer.from_pretrained(out/'delivered_model_qa')
model=AutoModelForQuestionAnswering.from_pretrained(out/'delivered_model_qa',attn_implementation='sdpa').to('cuda')
context='The office is in London. It opened in 2020.'
raw=Dataset.from_list([{'id':str(i),'question':q,'context':context,'answers':{'answer_start':[context.index(a)],'text':[a]}} for i,(q,a) in enumerate([('Where is the office?','London'),('When did the office open?','2020')])])
f=raw.map(lambda b:build_features(b,tok,cfg),batched=True,remove_columns=raw.column_names)
scores,preds=evaluate(model,raw,f,cfg,torch.device('cuda'))
save_json(out/'demo.json',{'context':context,'examples':[{'question':r['question'],'prediction':preds[r['id']]} for r in raw]})
print(preds)
