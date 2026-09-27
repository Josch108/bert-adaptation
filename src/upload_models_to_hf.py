"""Publish verified exports only; credentials are read securely, never saved here."""
from pathlib import Path
import argparse, getpass, hashlib, json, os, shutil
from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForQuestionAnswering

SPECS=[('agnews','bert-base-agnews-delivered','text-classification','AG News topic classification'),('ner','bert-base-cased-conll2003-ner','token-classification','CoNLL-2003 named entity recognition'),('pos','bert-base-uncased-ud-ewt-pos','token-classification','UD English EWT part-of-speech tagging'),('qa','bert-base-uncased-squad-qa','question-answering','SQuAD 1.1 extractive question answering')]
READ=lambda p:json.loads(p.read_text(encoding='utf-8'))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

DETAILS={
 'agnews':('fancyzhx/ag_news','12,000 training / 2,000 internal validation / 2,000 test texts. Stratified subsets; 161 duplicate official-train rows removed before sampling. Test uses the official test split.','Validation macro-F1; test accuracy and macro-F1.','BERT pooled output and a linear four-class head. Truncate inputs to 128 tokens.','English news classification into World, Sports, Business, Sci/Tech.','Inputs beyond 128 tokens are truncated. No evaluation on Spanish, other domains or other taxonomies.','AG News card currently lists the dataset license as unknown.'),
 'ner':('lhoestq/conll2003','Official 14,041 / 3,250 / 3,453 sentence splits. Exact sentence overlaps retained and audited: train/validation 129, train/test 78, validation/test 25.','Strict IOB2 micro entity F1 using seqeval 1.2.2. Full entity type and boundaries must match.','BERT-base-cased with a linear nine-label BIO head. Supervise only first subtokens; all other positions use -100. No truncation; first-subtoken word argmax, no BIO repair.','English news entity tagging for PER, ORG, LOC and MISC.','Small single-seed gap does not establish superiority. Repeated benchmark sentences limit independence. Not validated for Spanish or unseen entity categories.','The selected CoNLL dataset distribution does not clearly provide a dataset license on its card.'),
 'pos':('UniversalDependencies/UD_English-EWT','UD English EWT r2.15, official 12,544 / 2,001 / 2,077 sentence splits. Integer-ID syntactic words only; skip multiword surface rows and empty nodes. Exact overlaps retained: 36 / 47 / 27 across train/val, train/test, val/test.','Validation word accuracy; test word accuracy and macro-F1 over all 17 UPOS tags.','BERT-base-uncased with a linear 17-class head. First-subtoken supervision; other positions use -100. No truncation; dynamic padding.','English grammatical tagging of pretokenized words.','Accuracy gap is within the assignment seed-variation caution. Uncased input loses capitalization information. Not evaluated on Spanish or other domains.','UD English EWT r2.15 is distributed under CC BY-SA 4.0.'),
 'qa':('rajpurkar/squad','15,000 training / 2,000 internal validation / 2,000 held-out test questions. Reserve 15% of official training article titles for validation; test comes from official validation. No contexts overlap across selected subsets.','Normalized SQuAD 1.1 exact match and answer token F1 (0-100); validation F1 selects the checkpoint.','BERT predicts start/end positions. Max sequence 384, stride 128; preserve token_type_ids. Decode context-only top-20 start/end candidates, maximum answer length 30, best summed score across windows.','English answer extraction when an answer is present in a supplied context.','Does not generate new answers or evaluate abstention. Single seed and subset evaluation; SDPA backward can be nondeterministic. Not validated on Spanish or enterprise documents.','The SQuAD distribution is identified as CC BY-SA 4.0.')}

def card(task,title,repo,cfg,comparison,selected):
 dataset,split,metric,architecture,intended,limits,lic=DETAILS[task]
 url='https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.15' if task=='pos' else 'https://huggingface.co/datasets/'+dataset
 header=f'---\nlanguage: en\nlibrary_name: transformers\nbase_model: {cfg["model_id"]}\npipeline_tag: {next(s[2] for s in SPECS if s[0]==task)}\ntags:\n- bert\n- u2t01\n- academic\n'
 if task!='pos':header+=f'datasets:\n- {dataset}\n'
 header+='---\n'
 body=f'''# {title}

Selected method: **{selected['method']}**, epoch **{selected['best_epoch']}**. The choice was fixed on validation data before test evaluation. This is a measured academic adaptation of BERT, not a production-validated system.

## Training data and evaluation

{split}

{metric}

{architecture}

The table preserves native scales: AG News, NER and POS metrics are in [0, 1]; QA EM/F1 are in [0, 100]. Training seconds exclude validation, saving and final evaluation.

'''
 keys=['method','best_epoch','trainable_parameters','train_seconds']+[k for k in comparison[0] if k.startswith('test_') and k!='test_loss']
 body+='| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'
 for row in comparison:body+='| '+' | '.join(f'{row[k]:.6f}' if isinstance(row[k],float) else str(row[k]) for k in keys)+' |\n'
 body+=f'''
## Optimization and reproducibility

Three epochs, seed 42. AdamW: head LR 1e-3, trainable encoder LR 2e-5, weight decay 0.01, 10% linear warmup, clipping 1.0. Effective batch size {cfg['micro_batch']*cfg['accumulation']}. A partial method trains only the last two encoder layers and head; lower layers are frozen in evaluation mode. Frozen weights were checked for invariance. Full fine-tuning updates all parameters.

Base model revision: `{cfg['model_revision']}`. Dataset revision: `{cfg['dataset_revision']}`. Detailed configuration is in `training_config.json`; the complete comparison is in `evaluation.json`. The included experiment source and `requirements-lock.txt` document the original environment. Use a new output directory when reproducing. QA source includes the documented UTF-8 JSON read fix; it did not change any training weights.

Only one seed was evaluated. Small gaps may reflect initialization, dropout and ordering variability; no statistical significance is claimed. Training hardware: RTX 4060 Ti, CUDA bf16. Training runtime is not inference latency.

## Intended use and limitations

{intended}

{limits}

## Loading the delivered model

'''
 if task=='agnews':
  body+=f"```python\nfrom transformers import pipeline\nmodel = pipeline('text-classification', model='{repo}')\nprint(model('The team won the final match.', truncation=True, max_length=128))\n```\n"
 elif task in ('ner','pos'):
  body+=f"```python\nfrom transformers import AutoTokenizer, AutoModelForTokenClassification\nimport torch\nrepo = '{repo}'\ntokenizer = AutoTokenizer.from_pretrained(repo)\nmodel = AutoModelForTokenClassification.from_pretrained(repo).eval()\nwords = ['John', 'works', 'in', 'London', '.']\nbatch = tokenizer(words, is_split_into_words=True, return_tensors='pt')\nwith torch.no_grad():\n    ids = model(**batch).logits[0].argmax(-1).tolist()\nseen = set()\nfor i, word_id in enumerate(batch.word_ids()):\n    if word_id is not None and word_id not in seen:\n        print(words[word_id], model.config.id2label[ids[i]])\n        seen.add(word_id)\n```\n\nUse pretokenized words and first-subtoken predictions to match the benchmark. Generic aggregation pipelines may produce different results.\n"
 else:
  body+=f"```python\nfrom transformers import pipeline\nqa = pipeline('question-answering', model='{repo}')\nprint(qa(question='Where is the office?', context='The office is in London.',\n         max_seq_len=384, doc_stride=128, max_answer_len=30))\n```\n\nThis is an illustrative loading example. Exact benchmark reproduction uses the included experiment's cross-window decoder, not the pipeline's potentially different postprocessing.\n"
 body+=f'''
## Source terms and references

The upstream BERT checkpoints identify Apache-2.0 licensing. {lic} Dataset terms are separate from the base checkpoint license. This repository does not redistribute the training corpus. It preserves upstream attribution without asserting a new blanket license over all data sources.

- Base checkpoint: https://huggingface.co/{cfg['model_id']}
- Data source: {url}
- Devlin et al., BERT: https://arxiv.org/abs/1810.04805
- Transformers training: https://huggingface.co/docs/transformers/training
'''
 if task=='qa':body+='- Official SQuAD scorer: https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v1.1.py\n'
 if task=='ner':body+='- seqeval: https://github.com/chakki-works/seqeval\n'
 return header+body

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--staging',required=True);p.add_argument('--private',action='store_true');p.add_argument('--prompt-token',action='store_true');args=p.parse_args()
 root=Path(args.project).resolve();staging=Path(args.staging).resolve();staging.mkdir(parents=True,exist_ok=True)
 token=getpass.getpass('Hugging Face write token (hidden): ') if args.prompt_token else get_token()
 if not token:raise RuntimeError('No Hugging Face authentication. Run huggingface-cli login or use --prompt-token.')
 api=HfApi(token=token);account=api.whoami()['name'];print('Authenticated account:',account,flush=True)
 manifest_path=root/'publication_manifest.json'
 previous=READ(manifest_path) if manifest_path.exists() else {'models':[]}
 prior={m['task']:m for m in previous.get('models',[])}
 manifest={'account':account,'visibility':'private' if args.private else 'public','models':[]}
 prepared=[]
 for task,base,pipeline,title in SPECS:
  run=root/'runs'/f'{task}_verified';status=READ(run/'status.json');assert status['status'] in ('complete','completed') and not status.get('smoke',False)
  cfg=READ(run/'config.json');comparison=READ(run/'comparison.json');winner=READ(run/'selection.json')['winner'];selected=next(r for r in comparison if r['method']==winner)
  source=run/f'delivered_model_{task}';weight_hash=sha(source/'model.safetensors');assert weight_hash==sha(run/winner/'best_model'/'model.safetensors')
  folder=staging/task;folder.mkdir(exist_ok=True)
  for name in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']:
   assert (source/name).exists(),(task,name)
   shutil.copy2(source/name,folder/name)
  repo=f'{account}/{base}'
  # Preserve pre-existing unrelated repositories. Reuse only this recorded delivery.
  try:
   info=api.model_info(repo)
   if not (task in prior and prior[task].get('repo_id')==repo and prior[task].get('local_sha256')==weight_hash):
    repo+='-u2t01-verified'
    try:
     api.model_info(repo)
     if not (task in prior and prior[task].get('repo_id')==repo and prior[task].get('local_sha256')==weight_hash):raise RuntimeError('Target repository already exists without a matching delivery record: '+repo)
    except RepositoryNotFoundError:pass
  except RepositoryNotFoundError:pass
  (folder/'README.md').write_text(card(task,title,repo,cfg,comparison,selected),encoding='utf-8')
  write(folder/'training_config.json',cfg);write(folder/'evaluation.json',{'selected_method':winner,'comparison':comparison,'test_used_for_selection':False})
  shutil.copy2(root/'src'/f'{task}_experiment.py',folder/f'{task}_experiment.py')
  if task!='agnews':shutil.copy2(root/'src'/'agnews_experiment.py',folder/'agnews_experiment.py')
  shutil.copy2(run/'requirements-lock.txt',folder/'requirements-lock.txt')
  license_path=Path(__file__).with_name('BERT_BASE_LICENSE.txt')
  assert license_path.exists(), 'Missing upstream BERT license copy'
  shutil.copy2(license_path,folder/'BERT_BASE_LICENSE.txt')
  (folder/'NOTICE.md').write_text('Base checkpoint: '+cfg['model_id']+'\n\nThis model modifies pretrained BERT weights through task-specific adaptation. The original BERT source is Copyright 2018 The Google AI Language Team Authors and distributed under Apache License 2.0; see BERT_BASE_LICENSE.txt and https://github.com/google-research/bert. Dataset attribution and separate source terms are documented in README.md.\n',encoding='utf-8')
  for extra in ['utf8_recovery.json','utf8_fix.patch']:
   if (run/extra).exists():shutil.copy2(run/extra,folder/extra)
  AutoTokenizer.from_pretrained(folder,local_files_only=True)
  cls=AutoModelForSequenceClassification if task=='agnews' else AutoModelForQuestionAnswering if task=='qa' else AutoModelForTokenClassification
  model=cls.from_pretrained(folder,local_files_only=True);del model
  prepared.append((task,repo,folder,source,weight_hash))
  print('Preflight passed:',task,repo,flush=True)
 for task,repo,folder,source,weight_hash in prepared:
  api.create_repo(repo_id=repo,repo_type='model',private=args.private,exist_ok=True)
  record={'task':task,'repo_id':repo,'url':'https://huggingface.co/'+repo,'local_sha256':weight_hash,'verified':False}
  manifest['models'].append(record);write(manifest_path,manifest)
  commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=str(folder),commit_message='Publish verified U2T01 BERT model, tokenizer and experiment documentation')
  revision=commit.oid
  info=api.model_info(repo,revision=revision,files_metadata=True)
  remote=next(f for f in info.siblings if f.rfilename=='model.safetensors')
  lfs=remote.lfs;remote_hash=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None))
  if remote_hash!=weight_hash:
   downloaded=hf_hub_download(repo,'model.safetensors',revision=revision,token=token,cache_dir=str(staging/'verification-cache'))
   assert sha(Path(downloaded))==weight_hash
  downloaded_tokenizer=AutoTokenizer.from_pretrained(repo,revision=revision,token=token,cache_dir=str(staging/'verification-cache'))
  local_tok=AutoTokenizer.from_pretrained(folder)
  assert downloaded_tokenizer('John works in London.')['input_ids']==local_tok('John works in London.')['input_ids']
  config_file=hf_hub_download(repo,'config.json',revision=revision,token=token,cache_dir=str(staging/'verification-cache'))
  assert READ(Path(config_file))==READ(folder/'config.json')
  if not args.private:
   HfApi(token=False).model_info(repo,revision=revision)
  record.update({'revision':revision,'verified':True,'weight_hash_matches':True,'tokenizer_matches':True,'public_access_verified':not args.private})
  write(manifest_path,manifest)
  shutil.copy2(folder/'README.md',source/'README.md')
  print('Published and verified:',record['url'],revision,flush=True)
 print('Publication complete:',len(manifest['models']),'models',flush=True)
if __name__=='__main__':main()
