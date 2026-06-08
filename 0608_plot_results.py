#%% Imports & Config
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

DATA_PATH   = Path('data/UltrasonicData_comb_0604_Gain2.csv')
CONFIG_PATH = Path('models/jy_best_model.json')
SAVE_DIR    = Path('models')
OUT_SUMMARY = Path('results/0608_summary.png')
OUT_CM      = Path('results/0608_confusion_matrices.png')

OUT_SUMMARY.parent.mkdir(exist_ok=True)

with open(CONFIG_PATH, encoding='utf-8') as f:
    cfg = json.load(f)

FS       = cfg['inference_config']['sampling_rate']
IMG_SIZE = cfg['stft_config']['img_size']
NPERSEG  = cfg['stft_config']['nperseg']
NOVERLAP = cfg['stft_config']['noverlap']
NFFT     = cfg['stft_config']['nfft']
WINDOW   = cfg['stft_config']['window']

N_META_COLS = 5
BATCH_SIZE  = 32
VAL_RATIO   = 0.2
SEEDS       = [42, 55, 68, 81, 94]

CASES = [
    {'tag': 'SRB40160_RDB40179', 'filters': [('SRB-71C', '40160'), ('RDB-63FL', '40179')]},
    {'tag': 'SRB40160',          'filters': [('SRB-71C', '40160')]},
    {'tag': 'RDB40179',          'filters': [('RDB-63FL', '40179')]},
]

# ── Hardcoded results from training run ──────────────────────────────────────
RESULTS = {
    'SRB40160_RDB40179': {
        'val_acc': [0.9206, 0.9206, 0.9153, 0.9418, 0.9153],
        'f1':      [0.6154, 0.4444, 0.3333, 0.5600, 0.5000],
        'best_seed': 81,
    },
    'SRB40160': {
        'val_acc': [0.8807, 0.9174, 0.8624, 0.8807, 0.9083],
        'f1':      [0.5517, 0.5714, 0.4000, 0.5517, 0.5833],
        'best_seed': 55,
    },
    'RDB40179': {
        'val_acc': [0.9625, 0.9625, 0.9750, 0.9875, 0.9625],
        'f1':      [0.0000, 0.0000, 0.5000, 0.8000, 0.4000],
        'best_seed': 81,
    },
}

CASE_LABELS = {
    'SRB40160_RDB40179': 'Case 1\nSRB-71C/40160 + RDB-63FL/40179',
    'SRB40160':          'Case 2\nSRB-71C/40160',
    'RDB40179':          'Case 3\nRDB-63FL/40179',
}


#%% ── Figure 1: per-seed val_acc & F1 summary ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
x = np.arange(len(SEEDS))
w = 0.35

for ax, case in zip(axes, CASES):
    tag = case['tag']
    r   = RESULTS[tag]
    best_idx = SEEDS.index(r['best_seed'])

    bars_acc = ax.bar(x - w/2, r['val_acc'], w, label='Val Acc',  color='#4C72B0', alpha=0.85)
    bars_f1  = ax.bar(x + w/2, r['f1'],      w, label='F1(crack)', color='#DD8452', alpha=0.85)

    # highlight best seed
    bars_acc[best_idx].set_edgecolor('black'); bars_acc[best_idx].set_linewidth(2)
    bars_f1[best_idx].set_edgecolor('black');  bars_f1[best_idx].set_linewidth(2)

    for i, (a, f) in enumerate(zip(r['val_acc'], r['f1'])):
        ax.text(i - w/2, a + 0.005, f'{a:.3f}', ha='center', va='bottom', fontsize=7.5)
        ax.text(i + w/2, f + 0.005, f'{f:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'seed\n{s}' for s in SEEDS], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.set_title(f'{CASE_LABELS[tag]}\n'
                 f'mean Acc={np.mean(r["val_acc"]):.4f}  mean F1={np.mean(r["f1"]):.4f}',
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.axvline(best_idx, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(best_idx, 1.08, '★ best', ha='center', fontsize=8, color='black')

fig.suptitle('Multimodal Model — Val Acc & F1(crack) by Seed', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_SUMMARY, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved → {OUT_SUMMARY}')


#%% ── Load data & model for confusion matrices ───────────────────────────────
print('\nLoading data for confusion matrices ...')
df_all     = pd.read_csv(DATA_PATH, encoding='utf-8-sig', low_memory=False)
COL_NAME   = df_all.columns[0]
COL_CODE   = df_all.columns[1]
COL_LABEL  = df_all.columns[2]
signal_all = df_all.iloc[:, N_META_COLS:].values.astype(np.float32)

def to_signal_tensor(wav): return torch.from_numpy(wav).unsqueeze(0)

def to_stft_tensor(wav):
    _, _, Zxx = stft(wav, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                     nfft=NFFT, window=WINDOW)
    mag_db   = 20.0 * np.log10(np.abs(Zxx) + 1e-10)
    lo, hi   = mag_db.min(), mag_db.max()
    mag_norm = (mag_db - lo) / (hi - lo + 1e-10)
    img      = Image.fromarray((mag_norm * 255).astype(np.uint8))
    img      = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    t        = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
    return transforms.Normalize(mean=[0.5], std=[0.5])(t)

print('Precomputing tensors ...')
all_sig = [to_signal_tensor(w) for w in signal_all]
all_img = [to_stft_tensor(w)   for w in signal_all]
print('Done\n')


#%% Dataset & Model
class MultiModalDataset(Dataset):
    def __init__(self, sig_list, img_list, labels):
        self.sigs = sig_list; self.imgs = img_list
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.sigs[idx], self.imgs[idx], self.labels[idx]


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(True),
            nn.Conv1d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Sequential() if (stride == 1 and in_ch == out_ch) else \
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm1d(out_ch))
        self.relu = nn.ReLU(True)
    def forward(self, x): return self.relu(self.conv(x) + self.shortcut(x))

class Branch1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv1d(1,64,7,2,3,bias=False),nn.BatchNorm1d(64),nn.ReLU(True),nn.MaxPool1d(3,2,1))
        self.layer1 = nn.Sequential(BasicBlock1D(64,64),    BasicBlock1D(64,64))
        self.layer2 = nn.Sequential(BasicBlock1D(64,128,2), BasicBlock1D(128,128))
        self.layer3 = nn.Sequential(BasicBlock1D(128,256,2),BasicBlock1D(256,256))
        self.layer4 = nn.Sequential(BasicBlock1D(256,512,2),BasicBlock1D(512,512))
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Sequential(nn.Dropout(0.5),nn.Linear(512,128),nn.ReLU(True))
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.pool(x).squeeze(-1))

class Branch2D(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        for name, module in base.named_children():
            if name != 'fc': setattr(self, name, module)
        self.fc = nn.Sequential(nn.Dropout(0.5),nn.Linear(512,128),nn.ReLU(True))
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.fc(torch.flatten(self.avgpool(x),1))

class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_1d = Branch1D(); self.branch_2d = Branch2D()
        self.fusion = nn.Sequential(nn.Dropout(0.3),nn.Linear(256,64),nn.ReLU(True),nn.Linear(64,2))
    def forward(self, sig, img):
        return self.fusion(torch.cat([self.branch_1d(sig), self.branch_2d(img)], dim=1))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%% ── Figure 2: Confusion matrices for best seed per case ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, case in zip(axes, CASES):
    tag       = case['tag']
    best_seed = RESULTS[tag]['best_seed']
    ckpt_path = SAVE_DIR / f'0608_multimodal_{tag}_seed{best_seed}_best.pth'

    # re-derive same val split
    mask = pd.Series(False, index=df_all.index)
    for pname, pcode in case['filters']:
        mask |= (df_all[COL_NAME] == pname) & (df_all[COL_CODE].astype(str) == pcode)
    idx_case  = np.where(mask.values)[0]
    labels    = df_all.iloc[idx_case][COL_LABEL].values.astype(int)
    local_idx = np.arange(len(labels))

    import random
    random.seed(best_seed); np.random.seed(best_seed); torch.manual_seed(best_seed)
    _, val_sub = train_test_split(
        local_idx, test_size=VAL_RATIO, random_state=best_seed, stratify=labels
    )
    val_labels  = labels[val_sub]
    val_sig     = [all_sig[idx_case[i]] for i in val_sub]
    val_img     = [all_img[idx_case[i]] for i in val_sub]

    val_loader = DataLoader(
        MultiModalDataset(val_sig, val_img, val_labels),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model = MultiModalModel().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for sig, img, _ in val_loader:
            sig, img = sig.to(device), img.to(device)
            preds.extend(torch.softmax(model(sig, img), dim=1)[:, 1].cpu().numpy() >= 0.5)
    preds = np.array(preds, dtype=int)

    tp = int(((preds==1)&(val_labels==1)).sum())
    fp = int(((preds==1)&(val_labels==0)).sum())
    fn = int(((preds==0)&(val_labels==1)).sum())
    tn = int(((preds==0)&(val_labels==0)).sum())

    acc  = (tp + tn) / len(val_labels)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Normal', 'Pred Crack'],
                yticklabels=['Actual Normal', 'Actual Crack'],
                ax=ax)
    ax.set_title(
        f'{CASE_LABELS[tag]}\n'
        f'best seed={best_seed}  n={len(val_labels)}\n'
        f'Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}',
        fontsize=9,
    )
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    print(f'[{tag}] seed={best_seed}  TP={tp} FP={fp} FN={fn} TN={tn}  '
          f'Acc={acc:.4f}  F1={f1:.4f}')

fig.suptitle('Confusion Matrices — Best Seed per Case', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_CM, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved → {OUT_CM}')
