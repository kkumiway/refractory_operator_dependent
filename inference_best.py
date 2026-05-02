#%% Imports & Config
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.signal import stft
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from pathlib import Path

CONFIG_PATH  = Path('models/jy_best_model.json')    # 아키텍처 동일 → 같은 config 사용
WEIGHTS_PATH = Path('models/jy_best_model.pth')   # 학습 중 검증 성능 최고 체크포인트
DATA_PATH    = Path('data/UltrasonicData_combined_v1.csv')
N_META_COLS  = 5   # 제품명, 부호, 균열유무, 시간, 속도

with open(CONFIG_PATH, encoding='utf-8') as f:
    cfg = json.load(f)

stft_cfg   = cfg['stft_config']
infer_cfg  = cfg['inference_config']
model_cfg  = cfg['model_config']

FS          = infer_cfg['sampling_rate']
THRESHOLD   = infer_cfg['threshold']
IMG_SIZE    = stft_cfg['img_size']
NPERSEG     = stft_cfg['nperseg']
NOVERLAP    = stft_cfg['noverlap']
NFFT        = stft_cfg['nfft']
WINDOW      = stft_cfg['window']
NUM_CLASSES = model_cfg['num_classes']
SINGLE_CH   = model_cfg['use_single_channel']

print(f'Config 로드 완료: FS={FS:,} Hz, threshold={THRESHOLD}')
print(f'Weights: {WEIGHTS_PATH}')


#%% Data Loading
df = pd.read_csv(DATA_PATH, encoding='utf-8')

signal_np = df.iloc[:, N_META_COLS:].values.astype(np.float32)
labels    = df['균열유무'].values.astype(int)

print(f'데이터: {len(df)}행, 신호 길이: {signal_np.shape[1]} samples')
print(f'라벨 분포 — 정상(0): {(labels == 0).sum()}, 균열(1): {(labels == 1).sum()}')


#%% Preprocessing
def waveform_to_tensor(wav: np.ndarray) -> torch.Tensor:
    """파형 → STFT 스펙트로그램 → (1, IMG_SIZE, IMG_SIZE) float32 텐서"""
    _, _, Zxx = stft(
        wav, fs=FS,
        nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, window=WINDOW
    )
    mag_db = 20.0 * np.log10(np.abs(Zxx) + 1e-10)

    lo, hi   = mag_db.min(), mag_db.max()
    mag_norm = (mag_db - lo) / (hi - lo + 1e-10)

    img    = Image.fromarray((mag_norm * 255).astype(np.uint8))
    img    = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
    tensor = tensor.unsqueeze(0)
    tensor = transforms.Normalize(mean=[0.5], std=[0.5])(tensor)
    return tensor


#%% Model Loading
#
# best_model.pth 아키텍처
#   backbone : ResNet-18
#   conv1    : Conv2d(1, 64, 7×7)  ← 단채널 입력 (grayscale 스펙트로그램)
#   fc       : Sequential(
#                Dropout(0.5),
#                Linear(512 → 128),
#                ReLU,
#                Dropout(0.5),
#                Linear(128 → 2),   ← 정상/균열
#              )
#   저장 시점 : 학습 중 검증 손실(또는 정확도) 기준 best epoch
#
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load(WEIGHTS_PATH, map_location=device)

model = models.resnet18(weights=None)
if SINGLE_CH:
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

if 'fc.1.weight' in state_dict and 'fc.4.weight' in state_dict:
    in1  = state_dict['fc.1.weight'].shape[1]
    out1 = state_dict['fc.1.weight'].shape[0]
    out4 = state_dict['fc.4.weight'].shape[0]
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in1, out1),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(out1, out4),
    )
else:
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print(f'모델 로드 완료  |  device: {device}')
print(f'fc 구조: {model.fc}')


#%% Inference
BATCH_SIZE = 32

all_preds = []
all_probs = []

with torch.no_grad():
    for start in range(0, len(signal_np), BATCH_SIZE):
        batch_wav    = signal_np[start:start + BATCH_SIZE]
        batch_tensor = torch.stack([waveform_to_tensor(w) for w in batch_wav])
        batch_tensor = batch_tensor.to(device)

        logits = model(batch_tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = (probs[:, 1] >= THRESHOLD).astype(int)

        all_probs.extend(probs[:, 1].tolist())
        all_preds.extend(preds.tolist())

        if (start // BATCH_SIZE + 1) % 10 == 0:
            print(f'  {start + len(batch_wav)}/{len(signal_np)} 처리 완료...')

results_df = df[['제품명', '부호', '균열유무']].copy()
results_df['예측']       = all_preds
results_df['prob_crack'] = np.round(all_probs, 4)
results_df['correct']    = results_df['균열유무'] == results_df['예측']

print('\n추론 완료')


#%% Evaluation & Results
print('\n=== 샘플별 결과 (상위 20행) ===')
print(results_df.head(20).to_string(index=False))

print('\n=== 전체 정확도 ===')
acc = results_df['correct'].mean()
print(f'{acc:.4f}  ({results_df["correct"].sum()}/{len(results_df)})')

print('\n=== Classification Report ===')
print(classification_report(
    results_df['균열유무'], results_df['예측'],
    target_names=['정상(0)', '균열(1)']
))

print('=== Confusion Matrix ===')
cm    = confusion_matrix(results_df['균열유무'], results_df['예측'])
cm_df = pd.DataFrame(
    cm,
    index=['실제 정상', '실제 균열'],
    columns=['예측 정상', '예측 균열']
)
print(cm_df.to_string())

wrong = results_df[~results_df['correct']]
if len(wrong):
    print(f'\n=== 오분류 {len(wrong)}건 ===')
    print(wrong.to_string(index=False))
else:
    print('\n오분류 없음')
