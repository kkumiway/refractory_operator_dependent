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
SPLIT_DIR   = Path('data/splits')

with open(CONFIG_PATH, encoding='utf-8') as f:
    cfg = json.load(f)

FS       = cfg['inference_config']['sampling_rate']
IMG_SIZE = cfg['stft_config']['img_size']
NPERSEG  = cfg['stft_config']['nperseg']
NOVERLAP = cfg['stft_config']['noverlap']
NFFT     = cfg['stft_config']['nfft']
WINDOW   = cfg['stft_config']['window']

N_META_COLS = 5
EPOCHS      = 50
BATCH_SIZE  = 32
LR_BRANCH   = 1e-5
LR_FUSION   = 1e-4
VAL_RATIO   = 0.2
SEEDS       = [42, 55, 68, 81, 94]

CASES = [
    {'tag': 'SRB40160_RDB40179', 'filters': [('SRB-71C', '40160'), ('RDB-63FL', '40179')]},
    {'tag': 'SRB40160',          'filters': [('SRB-71C', '40160')]},
    {'tag': 'RDB40179',          'filters': [('RDB-63FL', '40179')]},
]


#%% Load & precompute both tensor types (once)
print(f'Loading {DATA_PATH.name} ...')
df_all     = pd.read_csv(DATA_PATH, encoding='utf-8-sig', low_memory=False)
COL_NAME   = df_all.columns[0]
COL_CODE   = df_all.columns[1]
COL_LABEL  = df_all.columns[2]
signal_all = df_all.iloc[:, N_META_COLS:].values.astype(np.float32)
print(f'Total rows: {len(df_all)}  |  signal length: {signal_all.shape[1]}')

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

print('Precomputing signal + STFT tensors ...')
t0 = time.time()
all_sig_tensors = [to_signal_tensor(w) for w in signal_all]
all_img_tensors = [to_stft_tensor(w)   for w in signal_all]
print(f'Done ({time.time()-t0:.1f}s)\n')


#%% Dataset
class MultiModalDataset(Dataset):
    def __init__(self, sig_list, img_list, labels):
        self.sigs   = sig_list
        self.imgs   = img_list
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.sigs[idx], self.imgs[idx], self.labels[idx]


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

class Branch1D(nn.Module):
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
        self.fc     = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(True))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.pool(x).squeeze(-1))
    def load_pretrained(self, path):
        res = self.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
        print(f'    Branch1D <- {Path(path).name}  (skipped {len(res.unexpected_keys)} keys)')

class Branch2D(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights='IMAGENET1K_V1')
        w = base.conv1.weight.mean(dim=1, keepdim=True)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            base.conv1.weight.copy_(w)
        for name, module in base.named_children():
            if name != 'fc':
                setattr(self, name, module)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(True))
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))
    def load_pretrained(self, path):
        res = self.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
        print(f'    Branch2D <- {Path(path).name}  (skipped {len(res.unexpected_keys)} keys)')

class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_1d = Branch1D()
        self.branch_2d = Branch2D()
        self.fusion = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(True), nn.Linear(64, 2),
        )
    def forward(self, sig, img):
        return self.fusion(torch.cat([self.branch_1d(sig), self.branch_2d(img)], dim=1))

def build_model(device, ckpt_1d, ckpt_2d):
    model = MultiModalModel()
    if ckpt_1d.exists():
        model.branch_1d.load_pretrained(ckpt_1d)
    else:
        print(f'    Branch1D — {ckpt_1d.name} not found, random init')
    if ckpt_2d.exists():
        model.branch_2d.load_pretrained(ckpt_2d)
    else:
        print(f'    Branch2D — {ckpt_2d.name} not found, ImageNet init only')
    return model.to(device)

def make_optimizer(model):
    return torch.optim.Adam([
        {'params': model.branch_1d.parameters(), 'lr': LR_BRANCH},
        {'params': model.branch_2d.parameters(), 'lr': LR_BRANCH},
        {'params': model.fusion.parameters(),    'lr': LR_FUSION},
    ])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}\n')


#%% Run
for case in CASES:
    tag = case['tag']

    mask = pd.Series(False, index=df_all.index)
    for pname, pcode in case['filters']:
        mask |= (df_all[COL_NAME] == pname) & (df_all[COL_CODE].astype(str) == pcode)
    idx_case    = np.where(mask.values)[0]
    labels      = df_all.iloc[idx_case][COL_LABEL].values.astype(int)
    sig_tensors = [all_sig_tensors[i] for i in idx_case]
    img_tensors = [all_img_tensors[i] for i in idx_case]
    n0, n1      = (labels == 0).sum(), (labels == 1).sum()

    print('=' * 60)
    print(f'CASE: {tag}  |  total={len(labels)}  normal={n0}  crack={n1}')
    print('=' * 60)

    cw   = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)
    print(f'class weights — normal: {cw[0]:.3f}  crack: {cw[1]:.3f}\n')

    local_idx = np.arange(len(labels))
    summary   = []

    HDR = f'  {"Seed":>5}  {"BestEp":>6}  {"ValAcc":>7}  {"ValLoss":>8}  {"F1":>7}'
    print(HDR); print('  ' + '-' * (len(HDR) - 2))

    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        tr_sub, val_sub = train_test_split(
            local_idx, test_size=VAL_RATIO, random_state=seed, stratify=labels
        )

        tr_loader = DataLoader(
            MultiModalDataset(
                [sig_tensors[i] for i in tr_sub],
                [img_tensors[i] for i in tr_sub],
                labels[tr_sub],
            ), batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            MultiModalDataset(
                [sig_tensors[i] for i in val_sub],
                [img_tensors[i] for i in val_sub],
                labels[val_sub],
            ), batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )

        ckpt_1d = SAVE_DIR / f'0608_1d_{tag}_seed{seed}.pth'
        ckpt_2d = SAVE_DIR / f'0608_2d_{tag}_seed{seed}.pth'
        model     = build_model(device, ckpt_1d, ckpt_2d)
        criterion = nn.CrossEntropyLoss(weight=cw_t)
        optimizer = make_optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_state = None; best_val_acc = 0.0; best_val_loss = float('inf')
        best_epoch = 0;    best_f1 = 0.0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for sig, img, y in tr_loader:
                sig, img, y = sig.to(device), img.to(device), y.to(device)
                optimizer.zero_grad(); criterion(model(sig, img), y).backward(); optimizer.step()
            scheduler.step()

            model.eval()
            loss_sum = correct = total = tp = fp = fn = 0
            with torch.no_grad():
                for sig, img, y in val_loader:
                    sig, img, y = sig.to(device), img.to(device), y.to(device)
                    out  = model(sig, img)
                    loss_sum += criterion(out, y).item() * len(y)
                    pred  = out.argmax(1)
                    correct += (pred == y).sum().item(); total += len(y)
                    tp += ((pred==1)&(y==1)).sum().item()
                    fp += ((pred==1)&(y==0)).sum().item()
                    fn += ((pred==0)&(y==1)).sum().item()

            val_acc = correct / total; val_loss = loss_sum / total
            prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc; best_val_loss = val_loss
                best_epoch = epoch;    best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        save_path = SAVE_DIR / f'0608_multimodal_{tag}_seed{seed}_best.pth'
        torch.save(best_state, save_path)
        print(f'  {seed:>5}  {best_epoch:>6}  {best_val_acc:>7.4f}  '
              f'{best_val_loss:>8.4f}  {best_f1:>7.4f}  → {save_path.name}')
        summary.append(dict(seed=seed, best_epoch=best_epoch,
                            val_acc=round(best_val_acc, 4), f1=round(best_f1, 4)))

    res = pd.DataFrame(summary)
    print(f'\n[{tag}] mean val_acc={res["val_acc"].mean():.4f}  '
          f'mean F1={res["f1"].mean():.4f}  '
          f'best seed={int(res.loc[res["val_acc"].idxmax(), "seed"])}\n')

print('All cases done.')
