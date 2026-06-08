#%% Imports & Config
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

DATA_PATH  = Path('data/UltrasonicData_comb_0604_Gain2.csv')
SAVE_DIR   = Path('models')
SPLIT_DIR  = Path('data/splits')

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

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


#%% Load & precompute tensors (once)
print(f'Loading {DATA_PATH.name} ...')
df_all     = pd.read_csv(DATA_PATH, encoding='utf-8-sig', low_memory=False)
COL_NAME   = df_all.columns[0]
COL_CODE   = df_all.columns[1]
COL_LABEL  = df_all.columns[2]
signal_all = df_all.iloc[:, N_META_COLS:].values.astype(np.float32)
print(f'Total rows: {len(df_all)}  |  signal length: {signal_all.shape[1]}')

def signal_to_tensor(wav): return torch.from_numpy(wav).unsqueeze(0)

print('Converting tensors ...')
t0 = time.time()
all_tensors = [signal_to_tensor(w) for w in signal_all]
print(f'Done ({time.time()-t0:.1f}s)\n')


#%% Dataset
class SignalDataset(Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels  = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.tensors[idx], self.labels[idx]


#%% Architecture
class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(True),
            nn.Conv1d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Sequential() if (stride == 1 and in_ch == out_ch) else \
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm1d(out_ch))
        self.relu = nn.ReLU(True)
    def forward(self, x): return self.relu(self.conv(x) + self.shortcut(x))

class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(1, 64, 7, 2, 3, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.MaxPool1d(3, 2, 1),
        )
        self.layer1 = nn.Sequential(BasicBlock1D(64, 64),     BasicBlock1D(64, 64))
        self.layer2 = nn.Sequential(BasicBlock1D(64, 128, 2),  BasicBlock1D(128, 128))
        self.layer3 = nn.Sequential(BasicBlock1D(128, 256, 2), BasicBlock1D(256, 256))
        self.layer4 = nn.Sequential(BasicBlock1D(256, 512, 2), BasicBlock1D(512, 512))
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(True),
            nn.Dropout(0.5), nn.Linear(128, 2),
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.pool(x).squeeze(-1))


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

        # save val split CSV (for reference / multimodal)
        val_csv = SPLIT_DIR / f'0608_val_{tag}_seed{seed}.csv'
        df_all.iloc[idx_case[val_sub]].to_csv(val_csv, index=False, encoding='utf-8-sig')

        tr_loader = DataLoader(
            SignalDataset([tensors[i] for i in tr_sub], labels[tr_sub]),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        )

        model     = ResNet1D().to(device)
        criterion = nn.CrossEntropyLoss(weight=cw_t)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        model.train()
        for epoch in range(1, EPOCHS + 1):
            for x, y in tr_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(); criterion(model(x), y).backward(); optimizer.step()
            scheduler.step()

        save_path = SAVE_DIR / f'0608_1d_{tag}_seed{seed}.pth'
        torch.save(model.state_dict(), save_path)
        print(f'  seed={seed}  train={len(tr_sub)}  val={len(val_sub)}  '
              f'→ {save_path.name}  val_csv={val_csv.name}')

    print()

print('All cases done.')
