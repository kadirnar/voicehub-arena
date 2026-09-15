"""Render current full-corpus plots from the published metrics snapshot."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,default=Path('hf-space/reports/full-metrics.json'))
p.add_argument('--output',type=Path,default=Path('hf-space/reports'))
a=p.parse_args()
OUT=a.output;OUT.mkdir(parents=True,exist_ok=True)
records=json.loads(a.source.read_text())['records']
assert len(records)>0 and all(r['summary']['scored']==1088 for r in records)
ordered=sorted(records,key=lambda r:(r['summary']['wer'],r['name'].casefold()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.spines.left':False,'svg.fonttype':'none'})

def save(fig,stem):
    for suffix in ('png','svg'):
        fig.savefig(OUT/f'{stem}.{suffix}',dpi=180,facecolor='white',bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(14,12),sharey=True)
y = np.arange(len(ordered))
for ax, metric, color in zip(axes,('wer','cer'),('#2563eb','#0f766e')):
    values = np.array([100*r['summary'][metric] for r in ordered])
    low = np.array([100*r['summary'][metric+'_ci95'][0] for r in ordered])
    high = np.array([100*r['summary'][metric+'_ci95'][1] for r in ordered])
    ax.errorbar(values,y,xerr=[values-low,high-values],fmt='o',color=color,
                ecolor=color,elinewidth=1,capsize=2,markersize=4,zorder=3)
    ax.set_xscale('log'); ax.set_xlim(max(low.min()*.7,.001),high.max()*1.35)
    ax.grid(axis='x',which='both',alpha=.14);ax.set_axisbelow(True)
    ax.set_title(metric.upper()+' (%)',loc='left',fontweight='bold')
    ax.set_xlabel('Lower is better · logarithmic axis')
    for i,v in enumerate(values):
        ax.text(1.025,i,f'{v:.2f}%',transform=ax.get_yaxis_transform(),va='center',fontsize=9)
axes[0].set_yticks(y,[r['name']+(' *' if r['quality_review'] else '') for r in ordered])
axes[0].invert_yaxis()
fig.suptitle('VoiceHub Arena · full Seed-TTS-Eval English',x=.03,ha='left',fontsize=19,fontweight='bold',y=.986)
fig.text(.03,.949,f'{len(records)} models × 1,088 texts  |  Whisper-large-v3  |  Corrected CosyVoice 3 full split',fontsize=11,color='#475569')
fig.text(.03,.018,'* High transcript error rates remain under review. Fixed provider voice/reference protocol; no speaker-SIM score.',fontsize=9,color='#475569')
fig.subplots_adjust(left=.17,right=.92,top=.915,bottom=.07,wspace=.30)
save(fig,'wer-cer-full')

speed = sorted(records,key=lambda r:r['summary']['rtf'])
fig,axes=plt.subplots(1,2,figsize=(14,12),sharey=True)
values=[r['summary']['rtf'] for r in speed]
memory=[r['summary']['peak_vram_mib']/1024 for r in speed]
axes[0].scatter(values,y,s=24,color='#7c3aed',zorder=3)
axes[0].set_xscale('log');axes[0].axvline(1,color='#94a3b8',linestyle='--',linewidth=1)
axes[0].set_title('Real-time factor (RTF)',loc='left',fontweight='bold')
axes[0].set_xlabel('Synthesis seconds / audio seconds · logarithmic axis')
axes[1].barh(y,memory,height=.55,color='#0891b2')
axes[1].set_xlim(0,max(memory)*1.2)
axes[1].set_title('Peak CUDA allocation (GiB)',loc='left',fontweight='bold')
axes[1].set_xlabel('Per-sample peak allocation; not total process VRAM')
for i,(v,m) in enumerate(zip(values,memory)):
    axes[0].text(1.025,i,f'{v:.3f}',transform=axes[0].get_yaxis_transform(),va='center',fontsize=9)
    axes[1].text(m+max(memory)*.015,i,f'{m:.2f}',va='center',fontsize=9)
for ax in axes:
    ax.grid(axis='x',which='both',alpha=.14);ax.set_axisbelow(True)
axes[0].set_yticks(y,[r['name'] for r in speed]);axes[0].invert_yaxis()
fig.suptitle('VoiceHub Arena · synthesis speed and memory',x=.03,ha='left',fontsize=19,fontweight='bold',y=.986)
fig.text(.03,.949,'A100-SXM4 40 GB  |  1,088 texts per model  |  Corrected CosyVoice 3 full split',fontsize=11,color='#475569')
fig.text(.03,.018,'RTF uses summed synthesis time / summed generated duration. Model loading, downloads and Whisper scoring are excluded.',fontsize=9,color='#475569')
fig.subplots_adjust(left=.17,right=.95,top=.915,bottom=.07,wspace=.3)
save(fig,'speed-memory-full')


for p in OUT.glob('*-full.svg'):
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
