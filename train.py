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

DATA_PATH   = Path('data/UltrasonicData_combined_v1.csv')
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
N_SPLITS    = 20
EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 1e-4
VAL_RATIO   = 0.2

print(f'Config: FS={FS:,} Hz, IMG_SIZE={IMG_SIZE}, NPERSEG={NPERSEG}')
print(f'학습 설정: {N_SPLITS}회 반복  |  {EPOCHS} epochs/run  |  LR={LR}  |  batch={BATCH_SIZE}')


#%% Data Loading
df        = pd.read_csv(DATA_PATH, encoding='utf-8')
signal_np = df.iloc[:, N_META_COLS:].values.astype(np.float32)
labels    = df['균열유무'].values.astype(int)

print(f'\n데이터: {len(df)}행  |  신호 길이: {signal_np.shape[1]} samples')
print(f'라벨 — 정상(0): {(labels==0).sum()}  균열(1): {(labels==1).sum()}')


#%% STFT Preprocessing — 전체 텐서 미리 계산
def waveform_to_tensor(wav: np.ndarray) -> torch.Tensor:
    _, _, Zxx = stft(wav, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                     nfft=NFFT, window=WINDOW)
    mag_db   = 20.0 * np.log10(np.abs(Zxx) + 1e-10)
    lo, hi   = mag_db.min(), mag_db.max()
    mag_norm = (mag_db - lo) / (hi - lo + 1e-10)
    img      = Image.fromarray((mag_norm * 255).astype(np.uint8))
    img      = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    t        = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
    return transforms.Normalize(mean=[0.5], std=[0.5])(t)

print('\nSTFT 전처리 중 (1회만 수행)...')
t0          = time.time()
all_tensors = [waveform_to_tensor(w) for w in signal_np]
print(f'완료 ({time.time()-t0:.1f}s)')


#%% Dataset
class SpecDataset(Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels  = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


#%% Model — ImageNet pretrained ResNet-18, 1ch 입력 적용
def build_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 3채널 conv1 가중치를 channel-average하여 1채널로 변환 (pretrained 정보 보존)
    w = model.conv1.weight.mean(dim=1, keepdim=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(w)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2),
    )
    return model.to(device)


#%% Class Weights (클래스 불균형 보정)
cw     = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cw_t   = torch.tensor(cw, dtype=torch.float32).to(device)

print(f'\ndevice: {device}')
print(f'class weights — 정상: {cw[0]:.3f}, 균열: {cw[1]:.3f}')
print(f'\n예상 소요 시간: {"GPU ~20분" if device.type=="cuda" else "CPU ~2-4시간"} '
      f'(GPU 사용 권장)\n')


#%% 20-Split Training
indices   = np.arange(len(labels))
results   = []
best_acc  = 0.0
best_path = SAVE_DIR / 'trained_best.pth'

HDR = f'{"Run":>4}  {"Seed":>5}  {"Ep":>4}  {"ValAcc":>7}  {"ValLoss":>8}  {"F1(crack)":>10}'
print('=' * len(HDR))
print(HDR)
print('=' * len(HDR))

for run in range(N_SPLITS):
    seed = run * 13 + 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    tr_idx, val_idx = train_test_split(
        indices, test_size=VAL_RATIO, random_state=seed, stratify=labels
    )

    tr_loader = DataLoader(
        SpecDataset([all_tensors[i] for i in tr_idx], labels[tr_idx]),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        SpecDataset([all_tensors[i] for i in val_idx], labels[val_idx]),
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
