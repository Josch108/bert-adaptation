from pathlib import Path
import argparse, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);args=p.parse_args()
root=Path(args.project);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
read=lambda f:json.loads(f.read_text(encoding='utf-8'))
methods={'feature_based':'Frozen + linear','frozen':'Frozen + linear','partial_finetuning':'Partial (top 2)','full_finetuning':'Full fine-tuning'}
for task,metric in [('agnews','val_macro_f1'),('ner','val_f1'),('pos','val_token_accuracy'),('qa','val_f1')]:
 run=root/'runs'/f'{task}_verified'
 fig,ax=plt.subplots(1,2,figsize=(8.3,2.1),layout='constrained')
 for row,color in zip(read(run/'comparison.json'),['#7652A8','#007F78']):
  h=read(run/row['method']/'history.json');ep=[v['epoch'] for v in h]
  ax[0].plot(ep,[v['train_loss'] for v in h],marker='o',ms=4,color=color,label=methods[row['method']])
  ax[1].plot(ep,[v[metric]*(1 if task=='qa' else 100) for v in h],marker='o',ms=4,color=color)
 for a in ax:
  a.set_xticks([1,2,3]);a.set_xlabel('Epoch',fontsize=8);a.tick_params(labelsize=8);a.spines[['top','right']].set_visible(False);a.grid(alpha=.15)
 ax[0].set_title('Training loss',fontsize=10,loc='left');ax[1].set_title('Validation selection score (0-100)',fontsize=10,loc='left');ax[0].legend(fontsize=7,frameon=False)
 fig.savefig(out/f'{task}_curves.png',dpi=190);plt.close(fig)
