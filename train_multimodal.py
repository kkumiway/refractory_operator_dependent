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

DATA_PATH    = Path('data/UltrasonicData_pretrain.csv')   # 90% 사전학습용
HOLDOUT_PATH = Path('data/UltrasonicData_holdout.csv')   # 10% hold-out (multi-modal 전용)
CONFIG_PATH = Path('models/jy_best_model.json')
SAVE_DIR    = Path('models')
CKPT_2D     = SAVE_DIR / 'trained_2d_best.pth'          # train_2d.py 결과
CKPT_1D     = SAVE_DIR / 'trained_1d_raw_best.pth'      # train_1d_raw.py 결과

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
LR_BRANCH   = 1e-5   # pretrained 브랜치 fine-tuning (느리게)
LR_FUSION   = 1e-4   # fusion head는 처음부터 학습 (빠르게)
VAL_RATIO   = 0.2

print(f'Config: FS={FS:,} Hz, IMG_SIZE={IMG_SIZE}')
print(f'학습 설정: {N_SPLITS}회 반복  |  {EPOCHS} epochs/run  |  batch={BATCH_SIZE}')
print(f'LR  —  branch: {LR_BRANCH}  fusion: {LR_FUSION}')


#%% Data Loading
df        = pd.read_csv(DATA_PATH, encoding='utf-8')
signal_np = df.iloc[:, N_META_COLS:].values.astype(np.float32)
labels    = df['균열유무'].values.astype(int)
SIG_LEN   = signal_np.shape[1]

print(f'\n데이터: {len(df)}행  |  신호 길이: {SIG_LEN} samples')
print(f'라벨 — 정상(0): {(labels==0).sum()}  균열(1): {(labels==1).sum()}')


#%% Preprocessing — 두 브랜치용 텐서를 각각 미리 계산
def to_stft_tensor(wav: np.ndarray) -> torch.Tensor:
    """원시 신호 → STFT 스펙트로그램 → (1, IMG_SIZE, IMG_SIZE)"""
    _, _, Zxx = stft(wav, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                     nfft=NFFT, window=WINDOW)
    mag_db   = 20.0 * np.log10(np.abs(Zxx) + 1e-10)
    lo, hi   = mag_db.min(), mag_db.max()
    mag_norm = (mag_db - lo) / (hi - lo + 1e-10)
    img      = Image.fromarray((mag_norm * 255).astype(np.uint8))
    img      = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    t        = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
    return transforms.Normalize(mean=[0.5], std=[0.5])(t)

def to_signal_tensor(wav: np.ndarray) -> torch.Tensor:
    """원시 신호 → (1, SIG_LEN) float32 텐서, 정규화 없음"""
    return torch.from_numpy(wav).unsqueeze(0)

print('\n전처리 중 (STFT + 원시 신호, 1회만 수행)...')
t0 = time.time()
sig_tensors  = [to_signal_tensor(w) for w in signal_np]
img_tensors  = [to_stft_tensor(w)   for w in signal_np]
print(f'완료 ({time.time()-t0:.1f}s)')


#%% Dataset
class MultiModalDataset(Dataset):
    def __init__(self, sig_list, img_list, labels):
        self.sigs   = sig_list
        self.imgs   = img_list
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sigs[idx], self.imgs[idx], self.labels[idx]


#%% Branch Models

# ── 2D Branch (ResNet-18 backbone, STFT 입력) ─────────────────────────────
class Branch2D(nn.Module):
    """ResNet-18 backbone + neck(512→128).  fc 직전 128-d feature 출력."""

    def __init__(self):
        super().__init__()
        base = models.resnet18(weights='IMAGENET1K_V1')
        # 3ch → 1ch: ImageNet 가중치를 채널 평균으로 보존
        w = base.conv1.weight.mean(dim=1, keepdim=True)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            base.conv1.weight.copy_(w)
        # ResNet-18의 모든 submodule을 그대로 복사 (fc 제외)
        for name, module in base.named_children():
            if name != 'fc':
                setattr(self, name, module)
        # fc를 512→128 neck으로 교체
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)   # (B, 128)

    def load_pretrained(self, path: Path):
        """train_2d.py 저장 가중치 로드 (fc 마지막 Linear 128→2는 무시)"""
        sd     = torch.load(path, map_location='cpu')
        result = self.load_state_dict(sd, strict=False)
        n_skip = len(result.unexpected_keys)
        print(f'  Branch2D  ← {path.name}  (skip {n_skip} keys: final classifier)')


# ── 1D Branch (ResNet1D backbone, 원시 신호 입력) ─────────────────────────
class BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class Branch1D(nn.Module):
    """ResNet1D backbone + neck(512→128).  fc 직전 128-d feature 출력."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlock1D(64,  64),  BasicBlock1D(64,  64))
        self.layer2 = nn.Sequential(BasicBlock1D(64,  128, stride=2), BasicBlock1D(128, 128))
        self.layer3 = nn.Sequential(BasicBlock1D(128, 256, stride=2), BasicBlock1D(256, 256))
        self.layer4 = nn.Sequential(BasicBlock1D(256, 512, stride=2), BasicBlock1D(512, 512))
        self.pool   = nn.AdaptiveAvgPool1d(1)
        # train_1d.py의 fc 키와 맞추기 위해 동일하게 fc로 정의 (마지막 Linear는 생략)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)   # (B, 128)

    def load_pretrained(self, path: Path):
        """train_1d.py 저장 가중치 로드 (fc 마지막 Linear 128→2는 무시)"""
        sd     = torch.load(path, map_location='cpu')
        result = self.load_state_dict(sd, strict=False)
        n_skip = len(result.unexpected_keys)
        print(f'  Branch1D  ← {path.name}  (skip {n_skip} keys: final classifier)')


# ── Multi-Modal Fusion Model ───────────────────────────────────────────────
class MultiModalModel(nn.Module):
    """
    Branch2D (STFT)  → 128-d ─┐
                                ├→ concat(256-d) → fusion MLP → 정상/균열
    Branch1D (signal) → 128-d ─┘
    """

    def __init__(self):
        super().__init__()
        self.branch_2d = Branch2D()
        self.branch_1d = Branch1D()
        self.fusion = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, sig: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        f1 = self.branch_1d(sig)               # (B, 128)
        f2 = self.branch_2d(img)               # (B, 128)
        return self.fusion(torch.cat([f1, f2], dim=1))  # (B, 2)


#%% Build Model & Load Pretrained Weights
def build_model(device: torch.device) -> MultiModalModel:
    model = MultiModalModel()
    print('사전학습 가중치 로드:')
    if CKPT_2D.exists():
        model.branch_2d.load_pretrained(CKPT_2D)
    else:
        print(f'  Branch2D  — {CKPT_2D.name} 없음, 랜덤 초기화 (ImageNet pretrained 유지)')
    if CKPT_1D.exists():
        model.branch_1d.load_pretrained(CKPT_1D)
    else:
        print(f'  Branch1D  — {CKPT_1D.name} 없음, 랜덤 초기화')
    return model.to(device)


def make_optimizer(model: MultiModalModel) -> torch.optim.Optimizer:
    """브랜치(fine-tuning)와 fusion(신규 학습)에 다른 LR 적용"""
    return torch.optim.Adam([
        {'params': model.branch_2d.parameters(), 'lr': LR_BRANCH},
        {'params': model.branch_1d.parameters(), 'lr': LR_BRANCH},
        {'params': model.fusion.parameters(),    'lr': LR_FUSION},
    ])


#%% Class Weights & Device
cw     = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cw_t   = torch.tensor(cw, dtype=torch.float32).to(device)

print(f'\ndevice: {device}')
print(f'class weights — 정상: {cw[0]:.3f}, 균열: {cw[1]:.3f}')

_m = MultiModalModel()
n_params = sum(p.numel() for p in _m.parameters())
print(f'총 파라미터: {n_params:,}')
del _m


#%% 20-Split Training
indices   = np.arange(len(labels))
results   = []
best_acc  = 0.0
best_path = SAVE_DIR / 'trained_multimodal_best.pth'

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
        MultiModalDataset(
            [sig_tensors[i] for i in tr_idx],
            [img_tensors[i] for i in tr_idx],
            labels[tr_idx],
        ),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        MultiModalDataset(
            [sig_tensors[i] for i in val_idx],
            [img_tensors[i] for i in val_idx],
            labels[val_idx],
        ),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model     = build_model(device)
    criterion = nn.CrossEntropyLoss(weight=cw_t)
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_state    = None
    best_val_acc  = 0.0
    best_val_loss = float('inf')
    best_epoch    = 0
    best_f1       = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        for sig, img, y in tr_loader:
            sig, img, y = sig.to(device), img.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(sig, img), y).backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        loss_sum = correct = total = tp = fp = fn = 0
        with torch.no_grad():
            for sig, img, y in val_loader:
                sig, img, y = sig.to(device), img.to(device), y.to(device)
                out  = model(sig, img)
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


#%% Summary — 20-split 결과
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


#%% Hold-out Evaluation — 학습에 한 번도 사용되지 않은 10% 데이터로 최종 평가
print('\n' + '=' * 55)
print('=== Hold-out 최종 평가 (UltrasonicData_holdout.csv) ===')

ho_df       = pd.read_csv(HOLDOUT_PATH, encoding='utf-8')
ho_signal   = ho_df.iloc[:, N_META_COLS:].values.astype(np.float32)
ho_labels   = ho_df['균열유무'].values.astype(int)

print(f'hold-out: {len(ho_df)}행  |  정상(0): {(ho_labels==0).sum()}  균열(1): {(ho_labels==1).sum()}')

ho_sig_tensors = [to_signal_tensor(w) for w in ho_signal]
ho_img_tensors = [to_stft_tensor(w)   for w in ho_signal]

ho_loader = DataLoader(
    MultiModalDataset(ho_sig_tensors, ho_img_tensors, ho_labels),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
)

best_model = build_model(device)
best_model.load_state_dict(torch.load(best_path, map_location=device))
best_model.eval()

ho_preds, ho_probs = [], []
with torch.no_grad():
    for sig, img, _ in ho_loader:
        sig, img = sig.to(device), img.to(device)
        out   = best_model(sig, img)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        ho_probs.extend(probs[:, 1].tolist())
        ho_preds.extend((probs[:, 1] >= 0.5).astype(int).tolist())

ho_correct = sum(p == t for p, t in zip(ho_preds, ho_labels))
ho_acc     = ho_correct / len(ho_labels)
tp = sum((p == 1 and t == 1) for p, t in zip(ho_preds, ho_labels))
fp = sum((p == 1 and t == 0) for p, t in zip(ho_preds, ho_labels))
fn = sum((p == 0 and t == 1) for p, t in zip(ho_preds, ho_labels))
prec   = tp / (tp + fp + 1e-9)
rec    = tp / (tp + fn + 1e-9)
ho_f1  = 2 * prec * rec / (prec + rec + 1e-9)

print(f'\nHold-out Accuracy : {ho_acc:.4f}  ({ho_correct}/{len(ho_labels)})')
print(f'Hold-out F1(crack): {ho_f1:.4f}')
print(f'Hold-out Precision: {prec:.4f}')
print(f'Hold-out Recall   : {rec:.4f}')
