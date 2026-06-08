#%% Imports & Config
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import stft
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

DATA_PATH   = Path('data/260501.xlsx')
CONFIG_PATH = Path('models/jy_best_model.json')
MODEL_PATH  = Path('models/trained_multimodal_best_0502.pth')
OUT_CSV     = Path('results/260501_predictions.csv')
OUT_CM      = Path('results/260501_confusion_matrix.png')

OUT_CSV.parent.mkdir(exist_ok=True)

with open(CONFIG_PATH, encoding='utf-8') as f:
    cfg = json.load(f)

FS       = cfg['inference_config']['sampling_rate']
IMG_SIZE = cfg['stft_config']['img_size']
NPERSEG  = cfg['stft_config']['nperseg']
NOVERLAP = cfg['stft_config']['noverlap']
NFFT     = cfg['stft_config']['nfft']
WINDOW   = cfg['stft_config']['window']

# xlsx 메타 컬럼 수 (0~30: 메타, 31~7962: 신호)
N_META_COLS = 31
BATCH_SIZE  = 32
THRESHOLD   = 0.5


#%% Data Loading
print('xlsx 로딩 중 (시간이 걸릴 수 있습니다)...')
df        = pd.read_excel(DATA_PATH)
signal_np = df.iloc[:, N_META_COLS:].values.astype(np.float32)   # (N, 7932)
labels_raw = df['Label'].values   # NaN 포함

print(f'전체: {len(df)}행  |  신호 길이: {signal_np.shape[1]} samples')
print(f'Label 분포 — 정상(0): {(labels_raw==0).sum():.0f}  '
      f'균열(1): {(labels_raw==1).sum():.0f}  '
      f'미검사(NaN): {pd.isna(labels_raw).sum()}')
print(f'이름 종류: {df["이름"].unique().tolist()}')


#%% Preprocessing
def to_stft_tensor(wav: np.ndarray) -> torch.Tensor:
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
    return torch.from_numpy(wav).unsqueeze(0)

print('\n전처리 중...')
sig_tensors = [to_signal_tensor(w) for w in signal_np]
img_tensors = [to_stft_tensor(w)   for w in signal_np]
print('완료')


#%% Dataset
class MultiModalDataset(Dataset):
    def __init__(self, sig_list, img_list):
        self.sigs = sig_list
        self.imgs = img_list
    def __len__(self): return len(self.sigs)
    def __getitem__(self, idx): return self.sigs[idx], self.imgs[idx]

loader = DataLoader(
    MultiModalDataset(sig_tensors, img_tensors),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
)


#%% Model
class Branch2D(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for name, module in base.named_children():
            if name != 'fc':
                setattr(self, name, module)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))

class BasicBlock1D(nn.Module):
    def __init__(self, ic, oc, s=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ic, oc, 3, s, 1, bias=False), nn.BatchNorm1d(oc), nn.ReLU(True),
            nn.Conv1d(oc, oc, 3, 1, 1, bias=False), nn.BatchNorm1d(oc),
        )
        self.shortcut = nn.Sequential() if (s == 1 and ic == oc) else \
            nn.Sequential(nn.Conv1d(ic, oc, 1, s, bias=False), nn.BatchNorm1d(oc))
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
        self.fc     = nn.Sequential(nn.Dropout(0.5), nn.Linear(512,128), nn.ReLU(True))
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.pool(x).squeeze(-1))

class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_2d = Branch2D()
        self.branch_1d = Branch1D()
        self.fusion = nn.Sequential(nn.Dropout(0.3), nn.Linear(256,64), nn.ReLU(True), nn.Linear(64,2))
    def forward(self, sig, img):
        return self.fusion(torch.cat([self.branch_1d(sig), self.branch_2d(img)], dim=1))


#%% Load & Inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = MultiModalModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f'\ndevice: {device}  |  모델: {MODEL_PATH.name}')

preds, probs_list = [], []
with torch.no_grad():
    for sig, img in loader:
        sig, img = sig.to(device), img.to(device)
        probs = torch.softmax(model(sig, img), dim=1).cpu().numpy()
        probs_list.extend(probs[:, 1].tolist())
        preds.extend((probs[:, 1] >= THRESHOLD).astype(int).tolist())

print(f'\n추론 완료  |  crack 예측: {sum(preds)}건 / {len(preds)}건')


#%% Results DataFrame
def classify(label, pred):
    if pd.isna(label): return 'Uninspected'
    label = int(label)
    if label == 1 and pred == 1: return 'TP'
    if label == 0 and pred == 1: return 'FP'
    if label == 1 and pred == 0: return 'FN'
    return 'TN'

results_df = df[['이름', 'Label', '측정 번호', '날짜 & 시간']].copy()
results_df['모델예측']   = preds
results_df['prob_crack'] = [round(p, 4) for p in probs_list]
results_df['result']     = [classify(l, p) for l, p in zip(labels_raw, preds)]

results_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f'CSV 저장: {OUT_CSV}')


#%% Metrics (Label 있는 142행 기준)
labeled = results_df[results_df['result'] != 'Uninspected']
tp = (labeled['result'] == 'TP').sum()
fp = (labeled['result'] == 'FP').sum()
fn = (labeled['result'] == 'FN').sum()
tn = (labeled['result'] == 'TN').sum()

acc  = (tp + tn) / len(labeled) if len(labeled) else 0
prec = tp / (tp + fp + 1e-9)
rec  = tp / (tp + fn + 1e-9)
f1   = 2 * prec * rec / (prec + rec + 1e-9)

print('\n' + '=' * 50)
print('=== 파괴 검사 결과와 비교 (Label 있는 행만) ===')
print(f'대상: {len(labeled)}행  |  실제 정상(0): {int((labeled["Label"]==0).sum())}  실제 균열(1): {int((labeled["Label"]==1).sum())}')
print(f'\nAccuracy  : {acc:.4f}  ({tp+tn}/{len(labeled)})')
print(f'Precision : {prec:.4f}  (crack 예측 중 실제 crack 비율)')
print(f'Recall    : {rec:.4f}  (실제 crack 중 맞힌 비율)')
print(f'F1(crack) : {f1:.4f}')
print(f'\nTP={tp}  FP={fp}  FN={fn}  TN={tn}')

wrong = labeled[labeled['result'].isin(['FP', 'FN'])]
if len(wrong):
    print(f'\n오분류 {len(wrong)}건:')
    print(wrong[['이름', 'Label', '모델예측', 'prob_crack', 'result', '날짜 & 시간']].to_string(index=False))


#%% Confusion Matrix (Label 있는 행)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ── left: vs. destructive inspection results ──────────────────────
cm_labeled = [[tn, fp], [fn, tp]]
sns.heatmap(cm_labeled, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Crack'],
            yticklabels=['Actual Normal', 'Actual Crack'],
            ax=axes[0])
axes[0].set_title(f'vs. Destructive Inspection Results (n={len(labeled)})\n'
                  f'Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}',
                  fontsize=10)
axes[0].set_ylabel('Actual (Destructive Inspection)'); axes[0].set_xlabel('Model Prediction')

# ── right: full prediction distribution ───────────────────────────
pred_counts = results_df['result'].value_counts()
colors = {'TP': '#2196F3', 'FP': '#FF9800', 'FN': '#F44336', 'TN': '#4CAF50', 'Uninspected': '#9E9E9E'}
bar_colors = [colors.get(k, '#999') for k in pred_counts.index]
axes[1].bar(pred_counts.index, pred_counts.values, color=bar_colors)
for i, (k, v) in enumerate(pred_counts.items()):
    axes[1].text(i, v + 10, str(v), ha='center', fontsize=10)
axes[1].set_title(f'Full Prediction Distribution (n={len(results_df)})', fontsize=10)
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig(OUT_CM, dpi=150)
plt.show()
print(f'그래프 저장: {OUT_CM}')