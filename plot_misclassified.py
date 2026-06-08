#%% Imports & Config
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import stft
from pathlib import Path

HOLDOUT_PRED_CSV = Path('results/holdout_predictions.csv')
HOLDOUT_DATA_CSV = Path('data/UltrasonicData_holdout.csv')
PRED_260501_CSV  = Path('results/260501_predictions.csv')
DATA_260501_CSV  = Path('data/260501.csv')
CONFIG_PATH      = Path('models/jy_best_model.json')
OUT_HOLDOUT      = Path('results/misclassified_holdout.png')
OUT_260501       = Path('results/misclassified_260501.png')
OUT_STATS_H      = Path('results/misclassified_holdout_stats.png')
OUT_STATS_2      = Path('results/misclassified_260501_stats.png')

OUT_HOLDOUT.parent.mkdir(exist_ok=True)

with open(CONFIG_PATH, encoding='utf-8') as f:
    cfg = json.load(f)

FS       = cfg['inference_config']['sampling_rate']
NPERSEG  = cfg['stft_config']['nperseg']
NOVERLAP = cfg['stft_config']['noverlap']
NFFT     = cfg['stft_config']['nfft']
WINDOW   = cfg['stft_config']['window']

HOLDOUT_META_COLS = 5
META_260501_COLS  = 31

COLORS = {'FP': '#FF9800', 'FN': '#F44336'}


#%% Helpers
def compute_stft_db(wav):
    f, t, Zxx = stft(wav, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                     nfft=NFFT, window=WINDOW)
    return f, t, 20.0 * np.log10(np.abs(Zxx) + 1e-10)


def plot_misclassified(signals, meta_list, dataset_label, out_path):
    n = len(signals)
    if n == 0:
        print(f'[{dataset_label}] No misclassified samples.')
        return

    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n), squeeze=False)

    for i, (wav, m) in enumerate(zip(signals, meta_list)):
        color = COLORS.get(m['result'], '#666')
        t_ms  = np.arange(len(wav)) / FS * 1000

        # --- raw signal ---
        ax = axes[i, 0]
        ax.plot(t_ms, wav, color=color, linewidth=0.6)
        ax.set_xlim(0, t_ms[-1])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(
            f"{m['name']}  |  Actual={int(m['actual'])}  "
            f"Pred={int(m['pred'])}  prob={m['prob']:.3f}  [{m['result']}]",
            fontsize=9, color=color,
        )
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2)

        # --- STFT spectrogram ---
        ax2 = axes[i, 1]
        f_arr, t_arr, mag_db = compute_stft_db(wav)
        im = ax2.pcolormesh(t_arr * 1000, f_arr / 1000, mag_db,
                            shading='gouraud', cmap='viridis')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Frequency (kHz)')
        ax2.set_title('STFT Spectrogram', fontsize=9)
        fig.colorbar(im, ax=ax2, label='dB', pad=0.02)
        for sp in ax2.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2)

    fp_patch = mpatches.Patch(color='#FF9800', label='FP — predicted crack, actually normal')
    fn_patch = mpatches.Patch(color='#F44336', label='FN — predicted normal, actually crack')
    fp_n = sum(1 for m in meta_list if m['result'] == 'FP')
    fn_n = sum(1 for m in meta_list if m['result'] == 'FN')
    fig.suptitle(
        f'{dataset_label} — Misclassified Samples  '
        f'(total={n}  FP={fp_n}  FN={fn_n})',
        fontsize=13, fontweight='bold',
    )
    fig.legend(handles=[fp_patch, fn_patch], loc='upper right', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out_path}')


def plot_name_stats(meta_list, dataset_label, out_path, total_counts):
    """Grouped bar chart + console table: FP/FN count per product name.
    total_counts: pd.Series mapping name -> total samples in full dataset."""
    if not meta_list:
        return

    stats = (
        pd.DataFrame(meta_list)
        .groupby(['name', 'result'])
        .size()
        .unstack(fill_value=0)
    )
    for col in ('FP', 'FN'):
        if col not in stats.columns:
            stats[col] = 0
    stats = stats[['FP', 'FN']]
    stats['total'] = stats['FP'] + stats['FN']
    stats = stats.sort_values('total', ascending=False)

    print(f'\n[{dataset_label}] Misclassified count by name:')
    print(stats.to_string())

    x = np.arange(len(stats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.4 + 1), 4))
    ax.bar(x - w / 2, stats['FP'], w, label='FP', color='#FF9800')
    ax.bar(x + w / 2, stats['FN'], w, label='FN', color='#F44336')
    for xi, (fp_v, fn_v) in enumerate(zip(stats['FP'], stats['FN'])):
        if fp_v: ax.text(xi - w / 2, fp_v + 0.05, str(fp_v), ha='center', fontsize=8)
        if fn_v: ax.text(xi + w / 2, fn_v + 0.05, str(fn_v), ha='center', fontsize=8)
    ax.set_xticks(x)
    tick_labels = [
        f"{name}\n(n={total_counts.get(name, '?')})" for name in stats.index
    ]
    ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Count')
    ax.set_title(
        f'{dataset_label} — Misclassified Count by Name  '
        f'(FP={int(stats["FP"].sum())}  FN={int(stats["FN"].sum())})',
        fontsize=12, fontweight='bold',
    )
    ax.legend()
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out_path}')


#%% ── Hold-out ──────────────────────────────────────────────────────────────
print('=== Hold-out ===')
holdout_pred = pd.read_csv(HOLDOUT_PRED_CSV, encoding='utf-8-sig')
holdout_data = pd.read_csv(HOLDOUT_DATA_CSV, encoding='utf-8')

wrong_h = holdout_pred[holdout_pred['result'].isin(['FP', 'FN'])]
print(f"FP={( holdout_pred['result']=='FP').sum()}  "
      f"FN={(holdout_pred['result']=='FN').sum()}  total={len(wrong_h)}")

signals_h, meta_h = [], []
for idx in wrong_h.index:
    wav = holdout_data.iloc[idx, HOLDOUT_META_COLS:].values.astype(np.float32)
    row = holdout_pred.iloc[idx]
    signals_h.append(wav)
    meta_h.append({
        'name':   str(row['제품명']),
        'actual': row['균열유무'],
        'pred':   row['예측'],
        'prob':   row['prob_crack'],
        'result': row['result'],
    })

plot_misclassified(signals_h, meta_h, 'Hold-out Dataset', OUT_HOLDOUT)
plot_name_stats(meta_h, 'Hold-out Dataset', OUT_STATS_H,
                total_counts=holdout_pred['제품명'].value_counts())


#%% ── 260501 ─────────────────────────────────────────────────────────────────
print('\n=== 260501 ===')
pred_260501 = pd.read_csv(PRED_260501_CSV, encoding='utf-8-sig')

print('Loading 260501.csv...')
data_260501 = pd.read_csv(DATA_260501_CSV, encoding='utf-8-sig')

wrong_2 = pred_260501[pred_260501['result'].isin(['FP', 'FN'])]
print(f"FP={(pred_260501['result']=='FP').sum()}  "
      f"FN={(pred_260501['result']=='FN').sum()}  total={len(wrong_2)}")

signals_2, meta_2 = [], []
for idx in wrong_2.index:
    wav = data_260501.iloc[idx, META_260501_COLS:].values.astype(np.float32)
    row = pred_260501.iloc[idx]
    signals_2.append(wav)
    meta_2.append({
        'name':   str(row['이름']),
        'actual': row['Label'],
        'pred':   row['모델예측'],
        'prob':   row['prob_crack'],
        'result': row['result'],
    })

plot_misclassified(signals_2, meta_2, '260501 Dataset', OUT_260501)
plot_name_stats(meta_2, '260501 Dataset', OUT_STATS_2,
                total_counts=pred_260501[pred_260501['result'] != 'Uninspected']['이름'].value_counts())

print('\nDone.')