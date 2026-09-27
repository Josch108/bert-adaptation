"""Build the English submission report from measured experiment artifacts."""
from pathlib import Path
import argparse, csv, json, html, re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

p = argparse.ArgumentParser()
p.add_argument('--project', required=True)
p.add_argument('--output', required=True)
args = p.parse_args()
ROOT = Path(args.project).resolve()
OUT = Path(args.output).resolve()
OUT.mkdir(parents=True, exist_ok=True)
read = lambda f: json.loads(f.read_text(encoding='utf-8'))
TASKS = ['agnews', 'ner', 'pos', 'qa']
NAMES = dict(zip(TASKS, ['AG News', 'Named entity recognition', 'Part-of-speech tagging', 'Extractive question answering']))
RUNS = {t: ROOT/'runs'/f'{t}_verified' for t in TASKS}
DATA = {t: read(RUNS[t]/'comparison.json') for t in TASKS}
CFG = {t: read(RUNS[t]/'config.json') for t in TASKS}
PUB = read(ROOT/'publication_manifest.json') if (ROOT/'publication_manifest.json').exists() else {'models': []}
PUBMAP = {m['task'].lower().replace(' ', ''): m for m in PUB.get('models', [])}
METHOD = {'feature_based':'Frozen + linear', 'frozen':'Frozen + linear', 'partial_finetuning':'Partial (top 2)', 'full_finetuning':'Full fine-tuning'}
VAL = {'agnews':'val_macro_f1', 'ner':'val_f1', 'pos':'val_token_accuracy', 'qa':'val_f1'}
PRIMARY = {'agnews':'test_macro_f1', 'ner':'test_f1', 'pos':'test_token_accuracy', 'qa':'test_f1'}
SCALE = {'agnews':100, 'ner':100, 'pos':100, 'qa':1}
for t in TASKS:
    assert read(RUNS[t]/'status.json')['status'] in ('complete','completed')
    assert len(DATA[t]) == 2

navy, teal, pale, gray = '#18354A', '#007F78', '#ECF4F5', '#526373'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyX', fontName='Helvetica', fontSize=9.3, leading=13.6, textColor=colors.HexColor(navy), spaceAfter=8))
styles.add(ParagraphStyle(name='SmallX', parent=styles['BodyX'], fontSize=8, leading=11, spaceAfter=5))
styles.add(ParagraphStyle(name='H1X', fontName='Helvetica-Bold', fontSize=23, leading=28, textColor=colors.HexColor(navy), spaceAfter=12))
styles.add(ParagraphStyle(name='H2X', fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor(teal), spaceBefore=10, spaceAfter=7))
styles.add(ParagraphStyle(name='KickerX', fontName='Helvetica-Bold', fontSize=8, leading=11, textColor=colors.HexColor(teal), spaceAfter=9))
styles.add(ParagraphStyle(name='CellX', parent=styles['SmallX'], fontSize=7.7, leading=10, spaceAfter=0))
styles.add(ParagraphStyle(name='CodeX', fontName='Courier', fontSize=7.2, leading=10.6, textColor=colors.HexColor(navy), spaceAfter=7, backColor=colors.HexColor(pale), borderPadding=6))
story, md = [], []
esc = lambda s: html.escape(str(s))
def para(text, small=False):
    story.append(Paragraph(text, styles['SmallX' if small else 'BodyX']))
    markdown=re.sub(r'<link href="([^"]+)"[^>]*>(.*?)</link>',lambda m:f'[{m[2]}]({m[1]})',text)
    md.extend([html.unescape(markdown.replace('<b>','**').replace('</b>','**').replace('<br/>','<br>\n')), ''])
def heading(title, level=2):
    story.append(Paragraph(esc(title), styles['H1X' if level==1 else 'H2X']))
    md.extend([('#' if level==1 else '##')+' '+title, ''])
def page(title, label):
    if story: story.append(PageBreak())
    story.append(Paragraph(label.upper(), styles['KickerX']))
    heading(title, 1)
def table(headers, rows, widths=None):
    data = [[Paragraph(esc(x), styles['CellX']) for x in headers]] + [[Paragraph(esc(x), styles['CellX']) for x in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(pale)),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor(teal)),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#DCE4E8')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.extend([t, Spacer(1,9)])
    md.extend(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'])
    md.extend('| '+' | '.join(str(v) for v in row)+' |' for row in rows)
    md.append('')
def code(text):
    story.append(Paragraph(esc(text).replace('\n','<br/>').replace(' ','&#160;'), styles['CodeX']))
    md.extend(['```text', text, '```', ''])
def curve(task):
    path=OUT/f'{task}_curves.png'
    assert path.exists(), 'Run render_curves.py first'
    story.append(Image(str(path),width=490,height=124))
    story.append(Spacer(1,7))
    md.extend([f'![{task} learning curves]({path.name})',''])
def epochs(task):
    rows=[]
    for r in DATA[task]:
        for h in read(RUNS[task]/r['method']/'history.json'):
            rows.append([METHOD[r['method']],h['epoch'],f"{h['train_loss']:.4f}",f"{h[VAL[task]]*SCALE[task]:.2f}",f"{h['train_seconds']:.1f}"])
    table(['Method','Epoch','Train loss','Val. score','Train sec.'],rows,[167,45,85,94,99])
def selected(t): return next(r for r in DATA[t] if r['selected'])
def refs_inline(numbers): return '['+', '.join(map(str,numbers))+']'

page('Adapting BERT for NLP Tasks', 'U2T01 / Final empirical report / 27 September 2026')
para('<b>Four tasks. Eight trained alternatives. Three adaptation strategies.</b> This study measures what freezing, partially adapting, and fully adapting BERT change in predictive quality and training cost. Each delivered model was selected on validation data before its held-out evaluation.')
heading('Main results')
rows=[]
for t in TASKS:
    for r in DATA[t]:
        metric={'agnews':'Macro-F1','ner':'Entity F1','pos':'Accuracy','qa':'Answer F1'}[t]
        rows.append([{'agnews':'AG News','ner':'NER','pos':'POS','qa':'QA'}[t], METHOD[r['method']], 'Yes' if r['selected'] else 'No',f"{metric}: {r[PRIMARY[t]]*SCALE[t]:.2f}",f"{r['trainable_parameters']/1e6:.3f}",f"{r['train_seconds']:.1f}"])
table(['Task','Method','Deliver','Test score /100','Trainable M','Train sec.'],rows,[47,124,43,120,77,79])
para('Metrics use a 0-100 display scale, but they measure different outcomes. Scores should be compared within a task, not used to rank task difficulty. Training time excludes validation, downloads, checkpoint saving and final evaluation.',True)
heading('What the measurements support')
para('Full fine-tuning was selected for AG News, NER and QA; partial fine-tuning was selected for POS. Relative to their trained alternatives, the selected models gained 15.68 macro-F1 points on AG News, 2.39 strict entity-F1 points on NER, 2.50 accuracy points on POS, and 21.53 answer-F1 points on QA.')
para('<b>Interpretation:</b> the large AG News and QA gaps support adapting BERT for these specific settings. NER and POS gaps are within the assignment\'s approximate 1-3 point seed-variation caution. With one seed per configuration, these are operational choices under the stated selection rule, not statistically established rankings.')
heading('Scope of the evidence')
para('All eight alternatives ran for three epochs with seed 42 on one NVIDIA GeForce RTX 4060 Ti. Saved configurations, histories, predictions, source snapshots and exported checkpoints support the reported measurements. No additional seeds or test-driven hyperparameter searches were performed. The previous draft numbers are superseded by this report.')

page('Study design and adaptation choices', '01 / Method')
para('BERT produces contextual representations by conditioning each token on surrounding tokens. The pretrained encoder is reused while the task head changes [1]. The experiment asks how much of this shared encoder should be updated, rather than training a language model from scratch.')
table(['Strategy','Trainable components','Implementation'],[
['Frozen features','Task head only','BERT remains in evaluation mode; no encoder gradients. AG News and POS use linear heads.'],
['Partial adaptation','Head + final 2 of 12 encoder layers','Embeddings and lower 10 layers are frozen; dropout in the frozen portion is disabled.'],
['Full adaptation','Embeddings, all encoder layers and head','All parameters update through the task loss.']], [104,155,231])
para('Frozen features are recomputed during training; there is no offline embedding cache. Frozen-parameter hashes were checked for invariance. This keeps the measured comparison tied to the implemented workflow rather than an idealized minimum feature-extraction cost.')
heading('Data and heads')
table(['Task / base checkpoint','Train / val. / test','Output head'],[
['AG News / uncased','12,000 / 2,000 / 2,000 texts','Pooled BERT output -> 4 class logits'],
['NER / cased','14,041 / 3,250 / 3,453 sentences','Per-token linear head -> 9 BIO labels'],
['POS / uncased','12,544 / 2,001 / 2,077 sentences','Per-token linear head -> 17 UPOS labels'],
['QA / uncased','15,000 / 2,000 / 2,000 questions','Per-token start and end scores']], [161,163,166])
para('AG News uses stratified subsets and removes normalized exact text duplicates before sampling (161 official-train rows excluded). NER and POS retain their official splits. Exact sentence overlaps were audited: NER train/val 129, train/test 78, val/test 25; POS 36, 47 and 27 respectively. Retaining these benchmark repetitions limits claims of wholly independent linguistic content. [2-4]')
para('QA reserves 15% of official training article titles for internal validation before sampling. The official validation split supplies the held-out test subset. No contexts overlap between the three selected subsets. These are custom evaluation subsets, not a claim of full official leaderboard performance. [5]')
heading('Optimization held constant within comparisons')
para('AdamW uses head learning rate 1e-3 and trainable encoder learning rate 2e-5, weight decay 0.01, 10% linear warmup, and gradient-norm clipping at 1.0. Separate parameter groups let a new head adapt faster while limiting updates to pretrained weights. The effective batch size is 32 for AG News and 16 for NER, POS and QA. CUDA bf16 was used. Exact package versions and model/dataset revisions are archived.')

page('Evaluation, alignment and selection', '02 / Validity checks')
table(['Task','Selection metric','Why this metric'],[
['AG News','Validation macro-F1','Gives each of the four topics equal weight; accuracy is also reported.'],
['NER','Strict IOB2 micro entity F1','Requires the correct entity type and complete word boundaries; O-token accuracy can hide errors.'],
['POS','Word accuracy','Matches the per-word tagging objective; macro-F1 exposes weaker or less frequent tags.'],
['QA','Normalized answer token F1','Measures partial answer overlap; exact match requires complete normalized agreement.']], [48,151,291])
para('The best validation epoch is retained, with the earliest epoch winning exact ties. Between methods, a tied selection score favors fewer trainable parameters. All selected checkpoints happened to be epoch 3. The method choice is persisted before test evaluation. Both selected and rejected alternatives are reported.')
heading('First-subtoken supervision for NER and POS')
para('Each word label is assigned only to its first WordPiece token. Continuation pieces, special tokens and padding receive -100 and do not contribute to cross-entropy. Loss is weighted by supervised words across accumulated batches. Dynamic padding is used; no NER/POS examples were truncated.')
table(['Task','Observed token','Original word','Training label'],[
['NER','[CLS]','-','-100 (ignored)'],['NER','la','lamb','O'],['NER','##mb','lamb','-100 (ignored)'],['POS','za','Zaman','PROPN'],['POS','##man','Zaman','-100 (ignored)']], [70,105,135,180])
para('These rows come from the saved alignment_check.csv files, not a hypothetical tokenizer example. Maximum observed subtoken lengths were 173 for NER and 337 for POS, below BERT\'s positional capacity.',True)
heading('QA uses span targets, not BIO labels')
para('Question and context are tokenized together with segment IDs. Overlapping 384-token windows use stride 128. Gold character offsets are mapped to start/end token positions; a window without the complete answer uses CLS for both targets. This is the task-specific equivalent of checking label alignment: a QA head does not use the NER/POS -100 word-label scheme.')
para('Decoding masks non-context tokens, considers the top 20 start and end candidates, rejects reversed or over-30-token spans, and chooses the highest summed score across windows. EM and F1 use SQuAD normalization and the maximum over annotated references [6]. Per-window loss and per-question evaluation must not be confused.')

page('AG News: adapting document representations', '03 / Topic classification')
a,b=DATA['agnews']
para('Working expectation: a frozen encoder should provide a useful baseline, while full adaptation may better separate nearby topics. Both alternatives use the same pooled-output linear classifier, so this comparison changes encoder adaptation rather than classifier capacity.')
table(['Method','Test accuracy','Test macro-F1','Training sec.'],[[METHOD[r['method']],f"{r['test_accuracy']*100:.2f}",f"{r['test_macro_f1']*100:.2f}",f"{r['train_seconds']:.1f}"] for r in DATA['agnews']], [184,102,102,102])
curve('agnews');epochs('agnews')
heading('Decision and error analysis')
para('Full fine-tuning was selected. It classified 1,850 of 2,000 test texts correctly versus 1,538 for the frozen alternative: a 15.60-point accuracy gain at 3.37 times the training time. The result supports the working expectation for this head, subset and configuration; it does not rule out stronger frozen-feature classifiers.')
para('Among the winner\'s 150 mistakes, 77 crossed the Business/Sci-Tech boundary (51 Business -> Sci/Tech and 26 in the reverse direction). Sports had 488/500 correct labels; Business had 435/500. Mixed business/technology content illustrates a difficult boundary without proving annotation error or a causal explanation.')
para('Inputs are capped at 128 tokens. Validation loss rose from epoch 2 to 3 while macro-F1 improved; checkpoint selection remained based on the predefined macro-F1 criterion. Loss also reflects confidence, so the metrics need not move together. The experiment does not evaluate Spanish or other news distributions.',True)

page('NER: entity boundaries and types', '04 / Named entity recognition')
para('Working expectation: adapting the final two layers may recover much of the entity-labeling benefit; full adaptation could add a smaller gain at higher cost. BERT-base-cased preserves case information that can be useful for entity names. This experiment did not isolate the effect of casing.')
table(['Method','Precision','Recall','Strict entity F1','Train sec.'],[[METHOD[r['method']],f"{r['test_precision']*100:.2f}",f"{r['test_recall']*100:.2f}",f"{r['test_f1']*100:.2f}",f"{r['train_seconds']:.1f}"] for r in DATA['ner']], [157,74,74,105,80])
curve('ner');epochs('ner')
heading('Decision and error analysis')
para('Full fine-tuning was selected by validation. It recovered 5,189 of 5,648 gold entities, versus 5,034 for partial adaptation, and produced 505 false-positive entities. Its 2.39-point test F1 advantage cost 3.03 times the training time. This gap is within the assignment\'s 1-3 point seed-variation warning; one run cannot establish that full adaptation is reliably better.')
para('Gold-entity errors for the selected model included 220 wrong types at exact boundaries, 157 overlapping boundary errors, and 82 missed entities. These three gold-error categories are disjoint; false-positive predictions are counted separately. MISC F1 was 81.46, compared with 96.14 for PER, indicating uneven performance across types.')
para('The primary scorer is seqeval strict IOB2. Secondary conlleval-style F1 is 87.96 for partial and 91.04 for full adaptation. Invalid BIO sequences are interpreted differently by these conventions, so strict scores are not directly interchangeable with other reported CoNLL scores. Predictions were not repaired. [7]',True)

page('POS: useful gains from limited adaptation', '05 / Part-of-speech tagging')
para('Working expectation: pretrained representations already encode useful grammar, but allowing two final layers to adapt may improve token decisions. Both methods use the same 17-class linear head. UD English EWT r2.15 CoNLL-U files are used directly for compatibility and provenance [4].')
table(['Method','Word accuracy','Macro-F1 (17 tags)','Training sec.'],[[METHOD[r['method']],f"{r['test_token_accuracy']*100:.2f}",f"{r['test_macro_f1']*100:.2f}",f"{r['train_seconds']:.1f}"] for r in DATA['pos']], [167,100,123,100])
curve('pos');epochs('pos')
heading('Decision and error analysis')
para('Partial adaptation was selected. It labeled 24,023 of 25,094 test words correctly, compared with 23,395 for the frozen encoder: 628 fewer mistakes and a 2.50-point accuracy gain. Training took 110.3 rather than 79.9 seconds (1.38 times as long). Macro-F1 improved by 3.24 points, from 86.51 to 89.75.')
para('The selected model\'s main confusion counts included PROPN -> NOUN (180), NOUN -> PROPN (109), ADJ -> NOUN (64), and NOUN -> ADJ (60). Uncased input removes a potentially useful cue for proper nouns, but this run does not test whether a cased model would solve those errors.')
para('Only integer-ID syntactic words are retained from CoNLL-U; multiword surface rows and empty nodes are skipped. Unknown or missing UPOS values raise an error. The 2.50-point accuracy gain is of the same order as the seed-variation caution; the selection is defensible under the fixed rule, but not a statistically robust superiority claim. Full fine-tuning was not tested for POS.',True)

page('QA: adapting BERT to extract answers', '06 / Question answering')
para('Working expectation: extracting a question-conditioned span may benefit more from updating the full encoder than from adapting two layers. Both methods start from the same uncased BERT checkpoint and a newly initialized span head, using the same questions and per-epoch batch order.')
table(['Method','Test exact match','Test token F1','Training sec.'],[[METHOD[r['method']],f"{r['test_exact_match']:.2f}",f"{r['test_f1']:.2f}",f"{r['train_seconds']:.1f}"] for r in DATA['qa']], [166,112,112,100])
curve('qa');epochs('qa')
heading('Decision and error analysis')
para('Full fine-tuning was selected. It gained 21.53 test F1 points and 25.80 exact-match points, with 2.77 times the training time. Its 73.60 EM corresponds to 1,472 fully matching answers out of 2,000 questions. There were 186 questions with zero token overlap and 342 with partial overlap.')
para('For a gold answer of "blazer", predicting "a compulsory blazer" has partial F1 but not exact match after normalization. Other errors select overly broad spans or a different passage fragment. These examples describe behavior; they are not evidence of a specific internal reasoning process.')
para('All positive windows were audited for context-only, fully covering gold spans. Token boundaries expanded 41/10/3 annotated spans in train/validation/test. At least one supervised span of up to 30 tokens exists for 14,965/15,000 training, 1,994/2,000 validation and all 2,000 test questions. SQuAD 1.1 does not evaluate abstention; this model is not validated on unanswerable questions.',True)

page('Reproducibility and delivered artifacts', '07 / Verification and publication')
para('The local project contains four training scripts, four executed notebooks, per-run configurations, package locks, data manifests, validation histories, test predictions, selected checkpoints and tokenizers. Model cards describe data, methods, metrics, intended use, limitations and references. The notebooks identify whether they train or inspect saved evidence; POS and QA display the recorded runs rather than pretending to retrain them.')
heading('Environment and rerun procedure')
para('Recorded runtime: Python 3.12.14; PyTorch 2.6.0+cu124; Transformers 4.49.0; Datasets 3.3.2; Hugging Face Hub 0.29.3. Use the requirements-lock.txt inside the corresponding verified run, a matching CUDA-enabled PyTorch installation, and the task guide. Use a new output directory to preserve the original evidence.')
code('python src/verify_pos_qa_artifacts.py .\npython src/pos_experiment.py --config runs/pos_verified/config.json --output runs/pos_replica')
para('For QA, construct Config from runs/qa_verified/config.json and pass it to run with a new output directory, as shown in QA_GUIDE.md. This preserves source revisions; the default CLI resolves current upstream revisions. Fixed seeds do not guarantee bitwise reproduction on other hardware or kernels. QA SDPA backward emitted a nondeterminism warning.')
heading('Evidence checks')
para('Saved predictions were used to independently recalculate POS and QA metrics. Each exported weight file was hashed against its selected checkpoint; validation selection, best epochs, test identities and executed notebooks were checked. QA export reload reproduced all 2,000 test predictions. Its final comparison initially failed because Windows decoded a UTF-8 JSON file as cp1252. Only that read was corrected; no training was repeated. The original snapshot, failed status, exact patch and recovery hashes are retained.')
heading('Hugging Face delivery status')
for task in TASKS:
    pub=PUBMAP.get(task)
    if pub and pub.get('verified'):
        para(f"<b>{NAMES[task]}:</b> <link href=\"{esc(pub['url'])}\" color=\"#007F78\">{esc(pub['repo_id'])}</link><br/>Verified commit: {esc(pub['revision'])}",True)
    else:
        para(f'<b>{NAMES[task]}:</b> local model and tokenizer ready; Hub publication not yet verified.',True)
para('publication_manifest.json records repository URLs, revisions and verification. Repository access is checked separately from local export; an existing local model is not evidence of successful publication.',True)
heading('Measured scope, not a production guarantee')
para('The study uses one seed, a fixed three-epoch budget and a limited search space. Different optimization schedules, cached frozen features or alternative heads may change the trade-off. The benchmarks are English and do not establish performance on private business data, Spanish, deployment latency, fairness or calibration. Training time is not inference latency.')

page('References and provenance', '08 / Sources and conclusions')
references=[
('Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.','https://arxiv.org/abs/1810.04805'),
('AG News dataset distribution used in this experiment (fancyzhx/ag_news).','https://huggingface.co/datasets/fancyzhx/ag_news'),
('CoNLL-2003 dataset distribution used in this experiment (lhoestq/conll2003).','https://huggingface.co/datasets/lhoestq/conll2003'),
('Universal Dependencies English EWT, release r2.15; CoNLL-U format.','https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.15'),
('Rajpurkar et al. SQuAD 1.1 dataset distribution.','https://huggingface.co/datasets/rajpurkar/squad'),
('SQuAD official v1.1 evaluation implementation.','https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v1.1.py'),
('seqeval sequence-labeling evaluation library (strict IOB2 configuration).','https://github.com/chakki-works/seqeval'),
('Hugging Face Transformers training documentation.','https://huggingface.co/docs/transformers/training'),
('Google BERT-base uncased checkpoint.','https://huggingface.co/google-bert/bert-base-uncased'),
('Google BERT-base cased checkpoint.','https://huggingface.co/google-bert/bert-base-cased')]
for i,(label,url) in enumerate(references,1):
    para(f'[{i}] {esc(label)}<br/><link href="{url}" color="#007F78">{url}</link>',True)
heading('Data and licensing notes')
para('The base BERT checkpoint cards identify Apache-2.0 licensing. The AG News distribution lists its license as unknown, and the CoNLL distribution does not clearly supply a dataset license on its card. UD English EWT distributes a CC BY-SA 4.0 license; the SQuAD distribution also identifies CC BY-SA 4.0. Dataset terms are not replaced by the base-model license. Hub packages contain trained weights, tokenizers and documentation, not copies of the training corpora. Source links and source revisions remain in the model cards and run configurations.',True)
heading('Final conclusion')
para('The adaptation ladder is a measured trade-off, not a fixed ranking. Full adaptation yielded large gains for news classification and extractive QA under this setup. Partial adaptation improved POS over frozen features with a modest time increase. For NER, the small observed gain from full adaptation came at roughly triple the training cost and remains uncertain under single-seed variability. The delivered choices follow validation evidence while retaining the rejected alternatives for inspection.')
para('This report was generated from the saved experiment JSON files. It supersedes the previous draft; earlier artifacts are retained in the project archive. The assignment brief U2T01.pdf defines the required tasks and deliverables.',True)

def footer(canvas, doc):
    w,h=A4
    canvas.setStrokeColor(colors.HexColor('#DCE4E8'));canvas.line(52,45,w-52,45)
    canvas.setFont('Helvetica',7.5);canvas.setFillColor(colors.HexColor(gray))
    canvas.drawString(52,31,'U2T01  |  BERT adaptation  |  Verified experimental results')
    canvas.drawRightString(w-52,31,str(doc.page))

doc=SimpleDocTemplate(str(OUT/'report.pdf'),pagesize=A4,rightMargin=52,leftMargin=52,topMargin=44,bottomMargin=58,title='Adapting BERT for NLP Tasks - Verified Empirical Study',author='U2T01 project',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'REPORT.md').write_text('\n'.join(md),encoding='utf-8')
print(json.dumps({'pdf':str(OUT/'report.pdf'),'markdown':str(OUT/'REPORT.md'),'publication_verified':len([m for m in PUB.get('models',[]) if m.get('verified')])}))
