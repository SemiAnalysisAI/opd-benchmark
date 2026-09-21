"""Plot measured development checkpoints; requires matplotlib (3.11.2 used)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args()
    runs = json.loads((args.root/'summary.json').read_text())
    baseline = next(r for r in runs if r['mode']=='eval' and r['status']=='complete')['evaluations'][0]
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    for ax, domain, title in zip(axes, ('countdown','graph_color'), ('Countdown','Graph Coloring')):
        teacher_points = {0: baseline[domain]['accuracy']*100}
        for r in runs:
            if r['mode']=='teacher' and r['domain']==domain and r['status']!='invalidated':
                teacher_points.update({e['step']:e[domain]['accuracy']*100 for e in r['evaluations']})
        xs=sorted(teacher_points)
        ax.plot(xs,[teacher_points[x] for x in xs],'-o',color='#156f63',label='Specialist teacher',linewidth=2)
        for r in runs:
            if r['mode']=='mopd' and r['status']!='invalidated' and r['evaluations']:
                es=r['evaluations']
                ax.plot([e['step'] for e in es],[e[domain]['accuracy']*100 for e in es],'-o',color='#a047a5',label=r['name'],linewidth=2)
        ax.set_title(title,loc='left',fontweight='bold')
        ax.set_xlabel('Optimizer updates')
        ax.set_ylim(0,100)
        ax.grid(axis='y',alpha=.2)
        ax.legend(frameon=False,loc='lower right')
    axes[0].set_ylabel('Development accuracy (%)')
    fig.suptitle('Tinker: independently trained teachers and routed MOPD student',fontsize=14,fontweight='bold',y=1.01)
    fig.text(.01,-.03,'512 fixed development puzzles/domain · 256-token cap · thinking disabled · single training seed\nTeacher and student updates are separate stages. Lines connect measured checkpoints; original teacher targets unavailable.',fontsize=9,color='#555555')
    fig.tight_layout()
    for ext in ('png','svg'):
        fig.savefig(args.root/f'learning-curves.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
