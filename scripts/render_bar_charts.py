"""Render shareable bar plots from the same immutable JSON used by the Space.

Run after any leaderboard data update. No inference, downloads, or score changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as DrawingPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import MaxNLocator

METRICS = {
    'wer': ('Word error rate', '%', 100, 2, 'wer_ci95'),
    'cer': ('Character error rate', '%', 100, 2, 'cer_ci95'),
    'rtf': ('Synthesis speed · real-time factor', '×', 1, 3, None),
    'memory': ('Peak CUDA allocation', 'GiB', 1 / 1024, 2, None),
}
COLORS = ['#32369a', '#0fa383', '#4385ee', '#bb754a', '#9771bd', '#317c39',
          '#c65b78', '#347f9e', '#827344', '#5868aa', '#00857a', '#a96797']
BG = '#fdfcf8'


def invalid(row):
    return row.get('score_status') == 'invalidated_by_implementation_bug'


def rounded_bar(ax, center, height, color, max_y, hatched=False):
    left, right = center - .39, center + .39
    rx, ry = .04, min(max_y * .018, height / 2)
    vertices = [(left, 0), (left, height-ry), (left, height), (left+rx, height),
                (right-rx, height), (right, height), (right, height-ry), (right, 0), (left, 0)]
    codes = [DrawingPath.MOVETO, DrawingPath.LINETO, DrawingPath.CURVE3, DrawingPath.CURVE3,
             DrawingPath.LINETO, DrawingPath.CURVE3, DrawingPath.CURVE3, DrawingPath.LINETO, DrawingPath.CLOSEPOLY]
    ax.add_patch(PathPatch(DrawingPath(vertices, codes), facecolor=color,
                          edgecolor='#99948b' if hatched else 'none',
                          hatch='///' if hatched else None, linewidth=.5, zorder=3))


def render(source, output):
    raw = source.read_bytes()
    data = json.loads(raw)
    assert len(data['table']) == 33 and data['samples_per_model'] == 1088
    output.mkdir(parents=True, exist_ok=True)
    colors = {r['model']: COLORS[i % len(COLORS)]
              for i, r in enumerate(sorted(data['table'], key=lambda r: r['model']))}
    colors.update(zip(('kokoro', 'supertonic', 'omnivoice', 'speecht5', 'f5tts', 'styletts2'), COLORS))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.spines.bottom': False})
    manifest = {'source_sha256': hashlib.sha256(raw).hexdigest(), 'samples_per_model': 1088,
                'axis': 'linear; zero baseline', 'confidence': 'Existing 95% prompt bootstrap intervals for WER/CER only', 'charts': []}
    for metric, (title, unit, factor, digits, ci_key) in METRICS.items():
        key = 'peak_vram_mib' if metric == 'memory' else metric
        ordered = sorted(data['table'], key=lambda r: (invalid(r), r[key], r['name']))
        for scope in ('best6', 'all'):
            rows = [r for r in ordered if not invalid(r)][:6] if scope == 'best6' else ordered
            values = [r[key] * factor for r in rows]
            upper = [r[ci_key][1] * factor if ci_key else value for r, value in zip(rows, values)]
            max_y = max(upper) * 1.28
            fig, ax = plt.subplots(figsize=(13.6 if scope == 'best6' else 38, 6.5), facecolor=BG)
            ax.set_facecolor(BG)
            fig.subplots_adjust(left=.06 if scope == 'best6' else .025, right=.988, top=.78, bottom=.23)
            fig.text(.045 if scope == 'best6' else .02, .94, title, fontsize=19, weight='bold', color='#282d3a')
            fig.text(.045 if scope == 'best6' else .02, .885,
                     f"{'Best 6 valid' if scope == 'best6' else 'All 33'} models · Seed-TTS-Eval English · 1,088 texts each · Lower is better", color='#7b8090', fontsize=11)
            fig.text(.982, .936, 'VoiceHub Arena', ha='right', fontsize=19, weight='bold', color='#343891')
            ax.set_xlim(-.6, len(rows)-.4)
            ax.set_ylim(0, max_y)
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#e6e2d8', linewidth=.9)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis='y', length=0, colors='#858997', pad=9)
            ax.set_xticks([])
            ax.text(-.018, 1.03, unit, transform=ax.transAxes, ha='right', color='#858997', fontsize=10)
            for i, (r, value, hi) in enumerate(zip(rows, values, upper)):
                color = '#c5c0b7' if invalid(r) else colors[r['model']]
                rounded_bar(ax, i, value, color, max_y, invalid(r))
                if ci_key:
                    low = r[ci_key][0] * factor
                    ax.errorbar(i, value, yerr=[[value-low], [hi-value]], fmt='none', ecolor='#383d48',
                                elinewidth=.9, alpha=.6, capsize=3.5, capthick=.9, zorder=4)
                marker = ' †' if invalid(r) else ' *' if r.get('quality_review') else ''
                ax.annotate(f'{value:.{digits}f}{unit if unit == "%" else ""}{marker}',
                            (i, hi), xytext=(0, 8), textcoords='offset points', ha='center', va='bottom',
                            fontsize=14 if scope == 'best6' else 10, weight='bold', color='#252935', zorder=5)
                initials = {'kokoro': 'K', 'supertonic': 'ST', 'omnivoice': 'OV', 'speecht5': 'S5',
                            'f5tts': 'F5', 'styletts2': 'S2', 'cosyvoice': 'C3', 'qwen3tts': 'Q3',
                            'mosstts': 'M', 'zonos2': 'Z2'}.get(r['model'], r['model'][:2].upper())
                ax.annotate(initials, (i, 0), xytext=(0, -2), textcoords='offset points', va='center', ha='center',
                            color='white', fontsize=10, weight='bold', annotation_clip=False,
                            bbox=dict(boxstyle='circle,pad=.65', facecolor=color, edgecolor=BG, linewidth=2), zorder=6)
                name = '\n'.join(textwrap.wrap(r['name'], width=22 if scope == 'best6' else 17))
                ax.annotate(name+marker, (i, 0), xytext=(0, -39), textcoords='offset points',
                            va='top', ha='center', fontsize=10 if scope == 'best6' else 8, color='#575c68', annotation_clip=False)
            note = ('Whiskers: 95% prompt bootstrap intervals.' if ci_key else 'Confidence intervals not measured. ' + ('Peak allocation, not total process VRAM.' if metric == 'memory' else 'Excludes loading, downloads and ASR.'))
            note += '  Linear axis starts at zero.'
            if any(invalid(r) for r in rows):
                note += '  † Archived, invalidated score; excluded from ranking.'
            if any(r.get('quality_review') and not invalid(r) for r in rows):
                note += '  * Quality under review.'
            fig.text(.045 if scope == 'best6' else .02, .055, note, fontsize=8.5, color='#858997')
            stem = f'{metric}-{scope}'
            for extension in ('png', 'svg'):
                fig.savefig(output/f'{stem}.{extension}', dpi=160, facecolor=BG,
                            metadata={'Title': title} if extension == 'svg' else {'Title': title, 'Description': note})
            plt.close(fig)
            manifest['charts'].append({'stem': stem, 'metric': key, 'unit': unit, 'scope': scope,
                                       'rows': [{'model': r['model'], 'value': r[key] * factor,
                                                 'ci95': [v * factor for v in r[ci_key]] if ci_key else None,
                                                 'invalidated': invalid(r)} for r in rows]})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f"Rendered {len(manifest['charts'])} charts in PNG + SVG from {manifest['source_sha256']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('hf-space/data/leaderboard.json'))
    parser.add_argument('--output', type=Path, default=Path('hf-space/reports/bars'))
    args = parser.parse_args()
    render(args.source, args.output)
