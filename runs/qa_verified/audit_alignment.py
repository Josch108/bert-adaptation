from pathlib import Path
import json
from qa_experiment import *
out=Path(__file__).parent/'alignment_audit'
cfg=Config(**json.loads((Path(__file__).parent/'runs/qa_verified/config.json').read_text()))
out.mkdir(exist_ok=True)
tok,raw,features=prepare(cfg,out)
audit={}
for split,fs in features.items():
 lookup={r['id']:r for r in raw[split]};positives=0;negatives=0;exact=0;expanded=[]
 for f in fs:
  r=lookup[f['example_id']];s=f['start_positions'];e=f['end_positions'];offsets=f['offset_mapping']
  if s==0:negatives+=1;continue
  positives+=1
  assert f['token_type_ids'][s]==1 and f['token_type_ids'][e]==1
  a,b=offsets[s][0],offsets[e][1]
  fully_covered=[(start,text) for start,text in zip(r['answers']['answer_start'],r['answers']['text']) if a<=start and b>=start+len(text)]
  assert fully_covered,(split,r['id'],a,b)
  if any(normalize(r['context'][a:b])==normalize(text) for _,text in fully_covered):exact+=1
  elif len(expanded)<5:expanded.append({'id':r['id'],'span':r['context'][a:b],'answers':[text for _,text in fully_covered]})
 audit[split]={'positive_windows':positives,'negative_windows':negatives,'all_positive_spans_fully_cover_reference':True,'normalized_exact_match_spans':exact,'token_boundary_expansion_examples':expanded}
save_json(Path(__file__).parent/'runs/qa_verified/alignment_audit.json',audit)
print(json.dumps(audit,indent=2))
