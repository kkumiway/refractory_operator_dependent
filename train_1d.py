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

DATA_PATH = Path('data/UltrasonicData_pretrain.csv')   # 90% 사전학습용
SAVE_DIR  = Path('models')

N_META_COLS = 5
N_SPLITS    = 20
EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 1e-4
VAL_RATIO   = 0.2

print(f'학습 설정: {N_SPLITS}회 반복  |  {EPOCHS} epochs/run  |  LR={LR}  |  batch={BATCH_SIZE}')


#%% Data Loading
df        = pd.read_csv(DATA_PATH, encoding='utf-8')
signal_np = df.iloc[:, N_META_COLS:].values.astype(np.float32)
labels    = df['균열유무'].values.astype(int)
SIG_LEN   = signal_np.shape[1]

print(f'\n데이터: {len(df)}행  |  신호 길이: {SIG_LEN} samples')
print(f'라벨 — 정상(0): {(labels==0).sum()}  균열(1): {(labels==1).sum()}')


#%% Preprocessing — per-sample z-score 정규화
def signal_to_tensor(wav: np.ndarray) -> torch.Tensor:
    """원시 신호 → (1, SIG_LEN) float32 텐서, z-score 정규화"""
    mean, std = wav.mean(), wav.std() + 1e-8
    return torch.from_numpy((wav - mean) / std).unsqueeze(0)  # (1, L)

print('\n신호 텐서 변환 중...')
t0          = time.time()
all_tensors = [signal_to_tensor(w) for w in signal_np]
print(f'완료 ({time.time()-t0:.1f}s)')


#%% Dataset
class SignalDataset(Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels  = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


#%% 1D ResNet Architecture
class BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    """ResNet-18 구조를 1D 신호에 맞게 변환한 모델"""
    def __init__(self, sig_len: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlock1D(64,  64),  BasicBlock1D(64,  64))
        self.layer2 = nn.Sequential(BasicBlock1D(64,  128, stride=2), BasicBlock1D(128, 128))
        self.layer3 = nn.Sequential(BasicBlock1D(128, 256, stride=2), BasicBlock1D(256, 256))
        self.layer4 = nn.Sequential(BasicBlock1D(256, 512, stride=2), BasicBlock1D(512, 512))
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)   # (B, 512)
        return self.fc(x)

    def extract_features(self, x) -> torch.Tensor:
        """multi-modal 용: fc 직전 128-d feature 반환"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc[0](x)   # Dropout
        x = self.fc[1](x)   # Linear 512→128
        x = self.fc[2](x)   # ReLU
        return x             # (B, 128)


def build_model(device: torch.device) -> ResNet1D:
    return ResNet1D(sig_len=SIG_LEN).to(device)


#%% Class Weights & Device
cw     = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cw_t   = torch.tensor(cw, dtype=torch.float32).to(device)

print(f'\ndevice: {device}')
print(f'class weights — 정상: {cw[0]:.3f}, 균열: {cw[1]:.3f}')

# 모델 파라미터 수 출력
_m = build_model(torch.device('cpu'))
n_params = sum(p.numel() for p in _m.parameters())
print(f'모델 파라미터: {n_params:,}')
del _m


#%% 20-Split Training
indices   = np.arange(len(labels))
results   = []
best_acc  = 0.0
best_path = SAVE_DIR / 'trained_1d_best.pth'

HDR = f'{"Run":>4}  {"Seed":>5}  {"Ep":>4}  {"ValAcc":>7}  {"ValLoss":>8}  {"F1(crack)":>10}'
print('\n' + '=' * len(HDR))
print(HDR)
print('=' * len(HDR))

for run in range(N_SPLITS):
    seed = run * 13 + 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    tr_idx, val_idx = train_test_split(
        indices, test_size=VAL_RATIO, random_state=seed, stratify=labels
    )

    tr_loader = DataLoader(
        SignalDataset([all_tensors[i] for i in tr_idx], labels[tr_idx]),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        SignalDataset([all_tensors[i] for i in val_idx], labels[val_idx]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model     = build_model(device)
    criterion = nn.CrossEntropyLoss(weight=cw_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_state    = None
    best_val_acc  = 0.0
    best_val_loss = float('inf')
    best_epoch    = 0
    best_f1       = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        loss_sum = correct = total = tp = fp = fn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y  = x.to(device), y.to(device)
                out   = model(x)
                loss_sum += criterion(out, y).item() * len(y)
                pred  = out.argmax(1)
                correct += (pred == y).sum().item()
                total   += len(y)
                tp += ((pred == 1) & (y == 1)).sum().item()
                fp += ((pred == 1) & (y == 0)).sum().item()
                fn += ((pred == 0) & (y == 1)).sum().item()

        val_acc  = correct / total
        val_loss = loss_sum / total
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc  = val_acc
            best_val_loss = val_loss
            best_epoch    = epoch
            best_f1       = f1
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    marker = ' ◀ best' if best_val_acc > best_acc else ''
    print(f'{run+1:>4}  {seed:>5}  {best_epoch:>4}  '
          f'{best_val_acc:>7.4f}  {best_val_loss:>8.4f}  {best_f1:>10.4f}{marker}')

    results.append(dict(
        run=run+1, seed=seed, best_epoch=best_epoch,
        val_acc=round(best_val_acc, 4),
        val_loss=round(best_val_loss, 4),
        f1_crack=round(best_f1, 4),
    ))

    if best_val_acc > best_acc:
        best_acc = best_val_acc
        torch.save(best_state, best_path)


#%% Summary
print('\n' + '=' * len(HDR))
print('=== 전체 결과 (val_acc 내림차순) ===')
res_df = pd.DataFrame(results).sort_values('val_acc', ascending=False).reset_index(drop=True)
print(res_df.to_string(index=False))

best_row = res_df.iloc[0]
print(f'\n최고 성능  →  Run {int(best_row["run"])} '
      f'(seed={int(best_row["seed"])}, epoch={int(best_row["best_epoch"])})')
print(f'  val_acc  : {best_row["val_acc"]:.4f}')
print(f'  f1_crack : {best_row["f1_crack"]:.4f}')
print(f'  저장 경로: {best_path}')
