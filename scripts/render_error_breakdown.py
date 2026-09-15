"""Plot the measured word-error components of the archived audit cohort."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(source, output):
    report=json.loads(source.read_text())
    selected={'vui':'Vui', 'conversationtts':'ConversationTTS', 'bark':'Bark Small',
              'vits':'VITS · MMS English','openvoice':'OpenVoice V2','voxcpm':'VoxCPM2'}
    rows=[r for r in report['models'] if r['model'] in selected]
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':11})
    fig,ax=plt.subplots(figsize=(12,6.5),facecolor='#fcfbf7')
    ax.set_facecolor('#fcfbf7');fig.subplots_adjust(left=.07,right=.98,top=.76,bottom=.2)
    bottom=np.zeros(len(rows))
    for field,label,color in [('word_substitutions','Wrong words','#37378d'),
                              ('word_deletions','Missing words','#e0a254'),
                              ('word_insertions','Extra words','#51a18c')]:
        values=np.array([100*r[field]/r['reference_words'] for r in rows])
        ax.bar(range(len(rows)),values,bottom=bottom,width=.68,color=color,label=label,zorder=3)
        bottom+=values
    assert np.allclose(bottom,[r['wer']*100 for r in rows],atol=1e-12)
    for i,v in enumerate(bottom): ax.text(i,v+.22,f'{v:.2f}%',ha='center',weight='bold',fontsize=13)
    ax.set_xticks(range(len(rows)),[selected[r['model']] for r in rows],fontsize=10)
    ax.set_ylim(0,14);ax.set_ylabel('Contribution to corpus WER (percentage points)')
    ax.yaxis.grid(True,color='#e1e0da',zorder=0);ax.tick_params(axis='both',length=0,pad=12,colors='#555963')
    for spine in ax.spines.values():spine.set_visible(False)
    ax.legend(ncol=3,loc='upper left',bbox_to_anchor=(-.01,1.17),frameon=False)
    fig.text(.07,.93,'Where the word errors come from',fontsize=21,weight='bold',color='#242736')
    fig.text(.07,.87,'Archived full runs · 1,088 English Seed-TTS-Eval texts per model · Whisper-large-v3',fontsize=11,color='#656975')
    fig.text(.07,.075,'Substitutions + deletions + insertions = WER. These counts do not by themselves identify the root cause.',fontsize=9,color='#656975')
    fig.text(.98,.075,'VoiceHub Arena',ha='right',fontsize=11,weight='bold',color='#37378d')
    output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['.png','.svg']:
        path=output.with_suffix(suffix);fig.savefig(path,dpi=160,facecolor=fig.get_facecolor())
        if suffix=='.svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();render(args.source,args.output)
