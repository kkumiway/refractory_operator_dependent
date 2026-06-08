#%% Imports & Config
import json
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

DATA_PATH   = Path('data/UltrasonicData_comb_0604_Gain2.csv')
CONFIG_PATH = Path('models/jy_best_model.json')
SAVE_DIR    = Path('models')

with open(CONFIG_PATH, encoding='utf-8') as f:
    cfg = json.load(f)

FS       = cfg['inference_config']['sampling_rate']
IMG_SIZE = cfg['stft_config']['img_size']
NPERSEG  = cfg['stft_config']['nperseg']
NOVERLAP = cfg['stft_config']['noverlap']
NFFT     = cfg['stft_config']['nfft']
WINDOW   = cfg['stft_config']['window']

N_META_COLS = 5
EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 1e-4
VAL_RATIO   = 0.2
SEEDS       = [42, 55, 68, 81, 94]

CASES = [
    {'tag': 'SRB40160_RDB40179', 'filters': [('SRB-71C', '40160'), ('RDB-63FL', '40179')]},
    {'tag': 'SRB40160',          'filters': [('SRB-71C', '40160')]},
    {'tag': 'RDB40179',          'filters': [('RDB-63FL', '40179')]},
]


#%% Load & precompute STFT tensors (once)
print(f'Loading {DATA_PATH.name} ...')
df_all     = pd.read_csv(DATA_PATH, encoding='utf-8-sig', low_memory=False)
COL_NAME   = df_all.columns[0]
COL_CODE   = df_all.columns[1]
COL_LABEL  = df_all.columns[2]
signal_all = df_all.iloc[:, N_META_COLS:].values.astype(np.float32)
print(f'Total rows: {len(df_all)}  |  signal length: {signal_all.shape[1]}')

def waveform_to_tensor(wav):
    _, _, Zxx = stft(wav, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                     nfft=NFFT, window=WINDOW)
    mag_db   = 20.0 * np.log10(np.abs(Zxx) + 1e-10)
    lo, hi   = mag_db.min(), mag_db.max()
    mag_norm = (mag_db - lo) / (hi - lo + 1e-10)
    img      = Image.fromarray((mag_norm * 255).astype(np.uint8))
    img      = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    t        = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
    return transforms.Normalize(mean=[0.5], std=[0.5])(t)

print('Precomputing STFT tensors ...')
t0 = time.time()
all_tensors = [waveform_to_tensor(w) for w in signal_all]
print(f'Done ({time.time()-t0:.1f}s)\n')


#%% Dataset
class SpecDataset(Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels  = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.tensors[idx], self.labels[idx]


#%% Model
def build_model(device):
    model = models.resnet18(weights='IMAGENET1K_V1')
    w = model.conv1.weight.mean(dim=1, keepdim=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(w)
    model.fc = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(128, 2),
    )
    return model.to(device)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}\n')


#%% Run
for case in CASES:
    tag = case['tag']

    mask = pd.Series(False, index=df_all.index)
    for pname, pcode in case['filters']:
        mask |= (df_all[COL_NAME] == pname) & (df_all[COL_CODE].astype(str) == pcode)
    idx_case = np.where(mask.values)[0]
    labels   = df_all.iloc[idx_case][COL_LABEL].values.astype(int)
    tensors  = [all_tensors[i] for i in idx_case]
    n0, n1   = (labels == 0).sum(), (labels == 1).sum()

    print('=' * 60)
    print(f'CASE: {tag}  |  total={len(labels)}  normal={n0}  crack={n1}')
    print('=' * 60)

    cw   = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)
    print(f'class weights — normal: {cw[0]:.3f}  crack: {cw[1]:.3f}\n')

    local_idx = np.arange(len(labels))

    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        tr_sub, val_sub = train_test_split(
            local_idx, test_size=VAL_RATIO, random_state=seed, stratify=labels
        )

        tr_loader = DataLoader(
            SpecDataset([tensors[i] for i in tr_sub], labels[tr_sub]),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        )

        model     = build_model(device)
        criterion = nn.CrossEntropyLoss(weight=cw_t)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        model.train()
        for epoch in range(1, EPOCHS + 1):
            for x, y in tr_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(); criterion(model(x), y).backward(); optimizer.step()
            scheduler.step()

        save_path = SAVE_DIR / f'0608_2d_{tag}_seed{seed}.pth'
        torch.save(model.state_dict(), save_path)
        print(f'  seed={seed}  train={len(tr_sub)}  val={len(val_sub)}  → {save_path.name}')

    print()

print('All cases done.')
