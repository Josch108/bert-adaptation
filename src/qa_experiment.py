"""Reproducible SQuAD 1.1 partial/full adaptation, isolated article splits."""
from __future__ import annotations
import argparse, collections, gc, hashlib, importlib.metadata, itertools, json, math, os, platform, re, shutil, string, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator, get_linear_schedule_with_warmup
from agnews_experiment import seed_everything, save_json, amp_context, synchronize

METHODS=('partial_finetuning','full_finetuning')
@dataclass
class Config:
    seed:int=42
    epochs:int=3
    train_size:int=15000
    eval_size:int=2000
    max_length:int=384
    stride:int=128
    micro_batch:int=4
    accumulation:int=4
    top_layers:int=2
    head_lr:float=1e-3
    backbone_lr:float=2e-5
    model_id:str='google-bert/bert-base-uncased'
    dataset_id:str='rajpurkar/squad'
    model_revision:str|None=None
    dataset_revision:str|None=None
    smoke:bool=False
    attention_backend:str='sdpa'
    gradient_checkpointing:bool=False
    n_best:int=20
    max_answer_length:int=30

# Equivalent to official SQuAD v1.1 evaluate-v1.1.py normalization/scoring.
# https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v1.1.py

def normalize(text):
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ', ''.join(c for c in text.lower() if c not in string.punctuation)).split())
def answer_score(pred, gold):
    p,g=normalize(pred).split(),normalize(gold).split()
    em=int(normalize(pred)==normalize(gold))
    common=sum((collections.Counter(p)&collections.Counter(g)).values())
    return em, (2*common/(len(p)+len(g)) if common else 0.)
def metrics(raw,predictions):
    scores=[(max(answer_score(predictions[r['id']],g)[0] for g in r['answers']['text']), max(answer_score(predictions[r['id']],g)[1] for g in r['answers']['text'])) for r in raw]
    return {'exact_match':100*float(np.mean([s[0] for s in scores])), 'f1':100*float(np.mean([s[1] for s in scores]))}

def build_features(batch,tokenizer,cfg):
    enc=tokenizer([q.lstrip() for q in batch['question']],batch['context'],truncation='only_second',max_length=cfg.max_length,stride=cfg.stride,return_overflowing_tokens=True,return_offsets_mapping=True,padding='max_length')
    samples=enc.pop('overflow_to_sample_mapping')
    assert 'token_type_ids' in enc, 'BERT question/context segment IDs are required'
    starts,ends,ids=[],[],[]
    for i,sample in enumerate(samples):
        offsets=enc['offset_mapping'][i]; seq=enc.sequence_ids(i)
        context=[j for j,s in enumerate(seq) if s==1]
        cls=enc['input_ids'][i].index(tokenizer.cls_token_id)
        start,end=cls,cls
        answers=batch['answers'][sample]
        for a,text in zip(answers['answer_start'],answers['text']):
            b=a+len(text)
            if context and offsets[context[0]][0]<=a and offsets[context[-1]][1]>=b:
                start=next(j for j in context if offsets[j][1]>a)
                end=next(j for j in reversed(context) if offsets[j][0]<b)
                break
        starts.append(start); ends.append(end); ids.append(batch['id'][sample])
        enc['offset_mapping'][i]=[o if seq[j]==1 else None for j,o in enumerate(offsets)]
    enc['start_positions']=starts; enc['end_positions']=ends; enc['example_id']=ids
    return enc

def postprocess(raw,features,start_logits,end_logits,n_best=20,max_answer_length=30):
    by_id=collections.defaultdict(list)
    for i,f in enumerate(features): by_id[f['example_id']].append(i)
    predictions={}
    for row in raw:
        best_score=-float('inf'); best=''
        for i in by_id[row['id']]:
            offsets=features[i]['offset_mapping']
            valid=np.array([o is not None and o[1]>o[0] for o in offsets])
            sl=np.where(valid,start_logits[i],-np.inf); el=np.where(valid,end_logits[i],-np.inf)
            si=np.argsort(-sl,kind='stable')[:n_best]; ei=np.argsort(-el,kind='stable')[:n_best]
            for s in si:
                for e in ei:
                    if not valid[s] or not valid[e] or e<s or e-s+1>max_answer_length: continue
                    score=float(sl[s]+el[e])
                    if score>best_score:
                        best_score=score; best=row['context'][offsets[s][0]:offsets[e][1]]
        predictions[row['id']]=best
    return predictions

def prepare(cfg,out):
    api=HfApi(); cfg.model_revision=cfg.model_revision or api.model_info(cfg.model_id).sha; cfg.dataset_revision=cfg.dataset_revision or api.dataset_info(cfg.dataset_id).sha
    save_json(out/'config.json',asdict(cfg))
    tok=AutoTokenizer.from_pretrained(cfg.model_id,revision=cfg.model_revision)
    raw=load_dataset(cfg.dataset_id,revision=cfg.dataset_revision)
    rng=np.random.default_rng(cfg.seed)
    titles=sorted(set(raw['train']['title'])); rng.shuffle(titles)
    val_titles=set(titles[:max(1,round(len(titles)*.15))])
    pools={'train':[i for i,t in enumerate(raw['train']['title']) if t not in val_titles], 'validation':[i for i,t in enumerate(raw['train']['title']) if t in val_titles], 'test':list(range(len(raw['validation'])))}
    indices={k:rng.choice(v, min(len(v), (32 if k=='train' else 16) if cfg.smoke else (cfg.train_size if k=='train' else cfg.eval_size)),replace=False).tolist() for k,v in pools.items()}
    data={k:raw['validation' if k=='test' else 'train'].select(ids) for k,ids in indices.items()}
    assert not (set(data['train']['title'])&set(data['validation']['title']))
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not(set(data[a]['context'])&set(data[b]['context'])),f'Context leakage {a}/{b}'
    features={k:ds.map(lambda b:build_features(b,tok,cfg),batched=True,remove_columns=ds.column_names,desc=f'Tokenize {k}') for k,ds in data.items()}
    save_json(out/'split_manifest.json',{'source_indices':indices,'source_splits':{'train':'train','validation':'train','test':'validation'},'example_ids':{k:ds['id'] for k,ds in data.items()},'sizes':{k:len(ds) for k,ds in data.items()},'features':{k:len(ds) for k,ds in features.items()},'validation_titles':sorted(val_titles),'context_overlap':0,'split_policy':'15% of official train article titles reserved for internal validation; official validation reserved for test; sample questions seed42'})
    coverage={}
    for k,fs in features.items():
        supported={f['example_id'] for f in fs if f['start_positions']!=0 and f['end_positions']-f['start_positions']+1<=30}
        coverage[k]={'examples_with_supervised_gold_span_at_most_30_tokens':len(supported),'total_examples':len(data[k]),'negative_windows':sum(f['start_positions']==0 for f in fs)}
    save_json(out/'span_coverage.json',coverage)
    return tok,data,features

def loader(features,cfg,shuffle=False):
    ds=features.remove_columns(['offset_mapping','example_id'])
    return DataLoader(ds,batch_size=cfg.micro_batch,shuffle=shuffle,collate_fn=default_data_collator,num_workers=0,generator=torch.Generator().manual_seed(cfg.seed))
@torch.no_grad()
def evaluate(model,raw,features,cfg,device):
    model.eval(); ss=[]; ee=[]; total=0.; count=0
    for batch in loader(features,cfg):
        batch={k:v.to(device) for k,v in batch.items()}
        with amp_context(device): output=model(**batch)
        n=len(batch['input_ids']); total+=output.loss.item()*n; count+=n
        ss.append(output.start_logits.float().cpu().numpy()); ee.append(output.end_logits.float().cpu().numpy())
    predictions=postprocess(raw,features,np.concatenate(ss),np.concatenate(ee),cfg.n_best,cfg.max_answer_length)
    return {**metrics(raw,predictions),'loss':total/count},predictions

def frozen_hash(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if not p.requires_grad: h.update(n.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def train(method,cfg,out,tok,raw,features,device):
    seed_everything(cfg.seed); folder=out/method; folder.mkdir()
    model=AutoModelForQuestionAnswering.from_pretrained(cfg.model_id,revision=cfg.model_revision,attn_implementation='sdpa')
    partial=method==METHODS[0]
    if partial:
        model.bert.requires_grad_(False)
        for layer in model.bert.encoder.layer[-cfg.top_layers:]: layer.requires_grad_(True)
    if partial:
        assert not any(p.requires_grad for p in model.bert.embeddings.parameters())
        assert all(p.requires_grad == (i>=12-cfg.top_layers) for i,layer in enumerate(model.bert.encoder.layer) for p in layer.parameters())
    before=frozen_hash(model) if partial else None
    model.to(device)
    if cfg.gradient_checkpointing and not partial: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    groups=[{'params':list(model.qa_outputs.parameters()),'lr':cfg.head_lr},{'params':[p for p in model.bert.parameters() if p.requires_grad],'lr':cfg.backbone_lr}]
    assert sum(p.numel() for g in groups for p in g['params'])==sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt=torch.optim.AdamW(groups,weight_decay=.01); dl=loader(features['train'],cfg,True)
    steps=math.ceil(len(dl)/cfg.accumulation)*cfg.epochs; sch=get_linear_schedule_with_warmup(opt,int(.1*steps),steps)
    history=[]; best=-1; trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,cfg.epochs+1):
        model.train()
        if partial:
            model.bert.embeddings.eval()
            for layer in model.bert.encoder.layer[:-cfg.top_layers]:layer.eval()
        synchronize(device); tic=time.perf_counter(); loss_sum=0; count=0; batch_no=0; iterator=iter(dl)
        while batches:=list(itertools.islice(iterator,cfg.accumulation)):
            opt.zero_grad(set_to_none=True); group_size=sum(len(b['input_ids']) for b in batches)
            for batch in batches:
                batch={k:v.to(device) for k,v in batch.items()}; n=len(batch['input_ids'])
                with amp_context(device): output=model(**batch); loss=output.loss*n/group_size
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward(); loss_sum+=output.loss.item()*n; count+=n; batch_no+=1
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); sch.step()
            if batch_no%100==0:
                progress={'method':method,'epoch':epoch,'batch':batch_no,'batches':len(dl)};save_json(out/'progress.json',progress);print(progress,flush=True)
        synchronize(device); train_seconds=time.perf_counter()-tic; tic=time.perf_counter()
        scores,preds=evaluate(model,raw['validation'],features['validation'],cfg,device)
        synchronize(device)
        row={'epoch':epoch,'train_loss':loss_sum/count,'train_seconds':train_seconds,'validation_seconds':time.perf_counter()-tic,**{'val_'+k:v for k,v in scores.items()}}
        history.append(row); save_json(folder/'history.json',history); print(method,row,flush=True)
        if scores['f1']>best:
            best=scores['f1']; best_epoch=epoch; model.save_pretrained(folder/'best_model');tok.save_pretrained(folder/'best_model');save_json(folder/'validation_predictions.json',preds)
    if partial: assert frozen_hash(model)==before
    summary={'method':method,'best_epoch':best_epoch,'best_val_f1':best,'trainable_parameters':trainable,'total_parameters':sum(p.numel() for p in model.parameters()),'frozen_parameters_verified':partial,'train_seconds':sum(r['train_seconds'] for r in history),'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None}
    save_json(folder/'summary.json',summary);del model,opt,sch;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return summary

def run(cfg,output,prepare_only=False):
    out=Path(output).resolve()
    if (out/'selection.json').exists() or (out/'partial_finetuning').exists(): raise FileExistsError('Refusing to overwrite experiment')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(8,os.cpu_count() or 1));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_json(out/'status.json',{'status':'preparing','smoke':cfg.smoke})
    tok,raw,features=prepare(cfg,out)
    if prepare_only: return
    save_json(out/'environment.json',{'python':platform.python_version(),'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'precision':'bf16' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32'})
    (out/'requirements-lock.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get('Name'))),encoding='utf-8')
    shutil.copy2(__file__,out/'qa_experiment.py');shutil.copy2(Path(__file__).with_name('agnews_experiment.py'),out/'agnews_experiment.py')
    try:
        summaries=[train(m,cfg,out,tok,raw,features,device) for m in METHODS]
        winner=max(summaries,key=lambda s:(s['best_val_f1'],-s['trainable_parameters']))['method']
        save_json(out/'selection.json',{'winner':winner,'criterion':'internal validation token F1; exact tie fewer parameters','test_used_for_selection':False,'single_seed':True,'smoke':cfg.smoke})
        for summary in summaries:
            folder=out/summary['method'];model=AutoModelForQuestionAnswering.from_pretrained(folder/'best_model',attn_implementation='sdpa').to(device)
            scores,preds=evaluate(model,raw['test'],features['test'],cfg,device)
            save_json(folder/'test_metrics.json',scores);save_json(folder/'test_predictions.json',preds)
            save_json(folder/'test_references.json',[dict(r) for r in raw['test']])
            errors=[{'id':r['id'],'question':r['question'],'context':r['context'],'references':r['answers']['text'],'prediction':preds[r['id']],'f1':max(answer_score(preds[r['id']],g)[1] for g in r['answers']['text'])} for r in raw['test']]
            save_json(folder/'error_analysis.json',{'zero_f1':sum(r['f1']==0 for r in errors),'partial_f1':sum(0<r['f1']<1 for r in errors),'examples':[r for r in errors if r['f1']<1][:20]})
            summary.update({'test_'+k:v for k,v in scores.items()});summary['selected']=summary['method']==winner
            del model;gc.collect();torch.cuda.empty_cache()
        save_json(out/'results.json',summaries);save_json(out/'comparison.json',summaries)
        if cfg.smoke:
            save_json(out/'status.json',{'status':'completed','smoke':True});return
        shutil.copytree(out/winner/'best_model',out/'delivered_model_qa',dirs_exist_ok=True)
        card={'config':asdict(cfg),'comparison':summaries,'selected_method':winner,'evaluation':'Internal article-disjoint validation for selection; official validation subset held out as test. One seed. EM/token F1 normalized as official SQuAD1.1; best of multiple references.','limitations':'English SQuAD1.1 academic subset; answers expected in context; no abstention; max answer30tokens; not validated for production or other domains.','sources':['https://huggingface.co/google-bert/bert-base-uncased','https://huggingface.co/datasets/rajpurkar/squad','https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v1.1.py']}
        (out/'delivered_model_qa'/'README.md').write_text('---\nlanguage: en\nbase_model: google-bert/bert-base-uncased\ndatasets:\n- rajpurkar/squad\npipeline_tag: question-answering\n---\n# Academic BERT QA\n\n```json\n'+json.dumps(card,indent=2)+'\n```\n',encoding='utf-8')
        model=AutoModelForQuestionAnswering.from_pretrained(out/'delivered_model_qa',attn_implementation='sdpa').to(device)
        scores,reloaded=evaluate(model,raw['test'],features['test'],cfg,device)
        assert reloaded==json.loads((out/winner/'test_predictions.json').read_text(encoding='utf-8'))
        save_json(out/'reload_verification.json',{'all_test_predictions_identical':True,'scores':scores})
        save_json(out/'status.json',{'status':'completed','smoke':cfg.smoke})
    except Exception as exc:
        save_json(out/'status.json',{'status':'failed','error':repr(exc)});raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    cfg=Config(smoke=args.smoke)
    if args.smoke:cfg.epochs=1
    run(cfg,args.output,args.prepare_only)
