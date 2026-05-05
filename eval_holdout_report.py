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

HOLDOUT_PATH = Path('data/UltrasonicData_holdout.csv')
CONFIG_PATH  = Path('models/jy_best_model.json')
MODEL_PATH   = Path('models/trained_multimodal_best.pth')
OUT_CSV      = Path('results/holdout_predictions.csv')
OUT_CM       = Path('results/confusion_matrix.png')

OUT_CSV.parent.mkdir(exist_ok=True)

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
THRESHOLD   = 0.5


#%% Data Loading
df        = pd.read_csv(HOLDOUT_PATH, encoding='utf-8')
signal_np = df.iloc[:, N_META_COLS:].values.astype(np.float32)
labels    = df['균열유무'].values.astype(int)

print(f'hold-out: {len(df)}행  |  정상(0): {(labels==0).sum()}  균열(1): {(labels==1).sum()}')


#%% Preprocessing (정규화 없음 — train_multimodal.py 현재 버전과 동일)
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

print('전처리 중...')
sig_tensors = [to_signal_tensor(w) for w in signal_np]
img_tensors = [to_stft_tensor(w)   for w in signal_np]


#%% Dataset
class MultiModalDataset(Dataset):
    def __init__(self, sig_list, img_list, labels):
        self.sigs   = sig_list
        self.imgs   = img_list
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.sigs[idx], self.imgs[idx], self.labels[idx]

loader = DataLoader(
    MultiModalDataset(sig_tensors, img_tensors, labels),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
)


#%% Model Definition
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
print(f'device: {device}  |  가중치: {MODEL_PATH.name}')

preds, probs_list = [], []
with torch.no_grad():
    for sig, img, _ in loader:
        sig, img = sig.to(device), img.to(device)
        probs = torch.softmax(model(sig, img), dim=1).cpu().numpy()
        probs_list.extend(probs[:, 1].tolist())
        preds.extend((probs[:, 1] >= THRESHOLD).astype(int).tolist())


#%% Build Results DataFrame
def classify(actual, pred):
    if actual == 1 and pred == 1: return 'TP'
    if actual == 0 and pred == 1: return 'FP'
    if actual == 1 and pred == 0: return 'FN'
    return 'TN'

results_df = df[['제품명', '부호', '균열유무']].copy()
results_df['예측']       = preds
results_df['prob_crack'] = [round(p, 4) for p in probs_list]
results_df['result']     = [classify(a, p) for a, p in zip(labels, preds)]

results_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f'\nCSV 저장: {OUT_CSV}')


#%% Metrics
tp = (results_df['result'] == 'TP').sum()
fp = (results_df['result'] == 'FP').sum()
fn = (results_df['result'] == 'FN').sum()
tn = (results_df['result'] == 'TN').sum()

acc  = (tp + tn) / len(results_df)
prec = tp / (tp + fp + 1e-9)
rec  = tp / (tp + fn + 1e-9)
f1   = 2 * prec * rec / (prec + rec + 1e-9)

print('\n' + '=' * 40)
print('=== Hold-out 평가 결과 ===')
print(f'Accuracy  : {acc:.4f}  ({tp+tn}/{len(results_df)})')
print(f'Precision : {prec:.4f}')
print(f'Recall    : {rec:.4f}')
print(f'F1(crack) : {f1:.4f}')
print(f'\nTP={tp}  FP={fp}  FN={fn}  TN={tn}')

wrong = results_df[results_df['result'].isin(['FP', 'FN'])]
if len(wrong):
    print(f'\n오분류 {len(wrong)}건:')
    print(wrong[['제품명', '부호', '균열유무', '예측', 'prob_crack', 'result']].to_string(index=False))


#%% Confusion Matrix
cm = [[tn, fp],
      [fn, tp]]

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True, fmt='d', cmap='Blues',
    xticklabels=['Predicted normal(0)', 'Predicted crack(1)'],
    yticklabels=['Actual normal(0)', 'Actual crack(1)'],
    ax=ax,
)
ax.set_title(f'Confusion Matrix  (Acc={acc:.3f}, F1={f1:.3f})', fontsize=12)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(OUT_CM, dpi=150)
plt.show()
print(f'Confusion matrix 저장: {OUT_CM}')