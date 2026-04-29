# PMD Fringe 이미지 ROI 좌표 검출 파이프라인

## 개요

카메라로 촬영한 PMD(Phase Measuring Deflectometry) Fringe 이미지에서 OLED 패널 영역의 사다리꼴 4점 좌표를 검출하는 딥러닝 기반 파이프라인입니다.

```
입력 : 흑백 fringe 이미지 (카메라로 tilt 촬영)
출력 : 사다리꼴 4점 좌표 [x1,y1, x2,y2, x3,y3, x4,y4] normalized [0, 1]
```

---

## 1. 문제 정의

### 1-1. 입력 이미지 특성

- 흑백 (grayscale) 이미지
- 카메라 촬영 각도로 인해 OLED 패널 영역이 **사다리꼴 형태**로 나타남
- 수직 / 수평 fringe 줄무늬 패턴 포함
- 패널 주변에 고정용 구조물 등 배경 요소 존재

### 1-2. 출력 좌표 정의

```
(x1, y1) : Top-Left      (TL)
(x2, y2) : Top-Right     (TR)
(x3, y3) : Bottom-Right  (BR)
(x4, y4) : Bottom-Left   (BL)

모든 좌표 : 이미지 크기로 정규화된 [0.0, 1.0]
```

### 1-3. 접근 방법

Pretrained CNN Backbone + Regression Head를 이용한 **8개 좌표 직접 예측 (Regression)** 방식을 채택합니다.

```
입력 이미지 (H, W, 1)
    → 3채널 복제 (H, W, 3)
    → ImageNet 정규화
    → CNN Backbone (ResNet-50)
    → Global Average Pooling
    → FC Head
    → Sigmoid
    → 8개 좌표 [0, 1]
```

---

## 2. 모델 구조

### 2-1. Backbone

| 항목 | 내용 |
|---|---|
| 모델 | ResNet-50 (ImageNet pretrained) |
| 입력 | (B, 3, 512, 512) |
| 출력 | Feature Map |

### 2-2. Head

```
Global Average Pooling
    → FC(512) → BN → ReLU → Dropout(0.3)
    → FC(256) → BN → ReLU
    → FC(8)   → Sigmoid
    → 출력: [x1,y1, x2,y2, x3,y3, x4,y4] ∈ [0, 1]
```

### 2-3. 흑백 입력 처리

PMD 이미지는 흑백 1채널이지만 ImageNet pretrained 모델은 3채널을 기대합니다.

```python
# 1채널 → 3채널 복제
gray_img = Image.open(path).convert("L")
img_3ch  = Image.merge("RGB", [gray_img, gray_img, gray_img])
```

### 2-4. Loss 함수

```python
loss = nn.SmoothL1Loss()(pred_coords, gt_coords)
```

### 2-5. Sigmoid 포화 방지

GT 좌표가 경계(0 또는 1)에 가까우면 gradient 소실 발생 가능성이 있습니다.

```
GT 좌표를 [0.05, 0.95] 범위로 클리핑하거나
이미지 패딩으로 사다리꼴이 경계에 닿지 않도록 데이터 생성
```

---

## 3. 데이터셋

### 3-1. 학습 데이터 종류

| 단계 | 데이터셋 | 이미지 수 | 용도 |
|---|---|---|---|
| Step 1 | SmartDoc-2015 Ch.1 | ~25,000장 | 기본 학습 |
| Step 2 | MIDV-2020 | ~72,000장 | 기본 학습 확장 |
| Step 3 | PMD 실제 이미지 | 50~100장 | Fine-tuning |
| Step 4 | 합성 PMD Fringe | 5,000~10,000장 | 재학습 |

### 3-2. CSV 라벨 형식 (공통)

```
image_path,       x1,    y1,    x2,    y2,    x3,    y3,    x4,    y4
images/001.jpg,   0.12,  0.08,  0.88,  0.10,  0.91,  0.87,  0.10,  0.85
```

- `image_path` : 데이터셋 root 기준 상대 경로
- 좌표 : 이미지 크기로 정규화된 `[0.0, 1.0]`

### 3-3. 폴더 구조

```
data/
├── smartdoc/
│   ├── frames/
│   │   ├── background01/
│   │   │   ├── datasheet001/
│   │   │   │   ├── frame_0001.jpeg
│   │   │   │   └── ...
│   │   │   └── ... (30종 문서)
│   │   └── ... (5개 배경)
│   └── metadata.csv
│
├── midv2020/
│   ├── images/
│   │   ├── esp_id/
│   │   │   ├── video/
│   │   │   ├── photo/
│   │   │   └── scan/
│   │   └── ... (10종 문서)
│   └── metadata.csv
│
├── pmd/
│   ├── images/
│   │   ├── vertical/
│   │   └── horizontal/
│   └── metadata.csv
│
├── synth_pmd/
│   ├── images/
│   └── metadata.csv
│
└── splits/
    ├── smartdoc_train.csv
    ├── smartdoc_val.csv
    ├── midv2020_train.csv
    ├── midv2020_val.csv
    ├── pmd_train.csv
    ├── pmd_val.csv
    ├── synth_pmd_train.csv
    └── synth_pmd_val.csv
```

### 3-4. Train / Val 분할

| 데이터셋 | Train | Val |
|---|---|---|
| SmartDoc-2015 | 90% | 10% |
| MIDV-2020 | 90% | 10% |
| PMD 실제 | 80% | 20% |
| 합성 PMD | 90% | 10% |

### 3-5. 데이터 다운로드

**SmartDoc-2015**
```
URL  : https://zenodo.org/record/1230218
파일 : frames.tar.gz
압축 해제 : tar -xzf frames.tar.gz -C data/smartdoc/frames/
```

**MIDV-2020**
```
URL  : ftp://smartengines.com/midv-2020
       또는 http://l3i-share.univ-lr.fr
GT   : JSON → 공통 CSV 형식으로 변환 필요
```

**PMD 실제 이미지**
```
직접 촬영 후 LabelMe / CVAT로 4점 polygon 라벨링
저장 형식 : 공통 CSV (normalized [0,1])
```

**합성 PMD Fringe**
```
수식 기반 생성 (아래 합성 데이터 생성 참고)
생성 시 metadata.csv 자동 저장
```

### 3-6. 합성 PMD Fringe 생성 방법

```python
# 수직 fringe
I(x, y) = A + B * cos(2π * f * x + φ)

# 수평 fringe
I(x, y) = A + B * cos(2π * f * y + φ)

# 파라미터 랜덤 샘플링
A : 100~180      (배경 밝기)
B : 30~80        (진폭)
f : 1/30~1/8     (공간 주파수, px⁻¹)
φ : 0~2π         (위상)

# 사다리꼴 면적비: 50%~90%
# 면적 계산: Shoelace formula
Area = 0.5 * |x1(y2-y4) + x2(y3-y1) + x3(y4-y2) + x4(y1-y3)|
```

---

## 4. Augmentation

### 4-1. 단계별 Augmentation 설정

`torchvision.transforms.v2` 기반으로 구현합니다.

| Augmentation | Step 1 | Step 2 | Step 3 | Step 4 |
|---|---|---|---|---|
| Stripe Overlay | p=0.4 | p=0.5 | ❌ | ❌ |
| Grayscale | p=0.5 | p=0.5 | p=1.0 | p=1.0 |
| HorizontalFlip | p=0.5 | p=0.5 | p=0.5 | p=0.5 |
| Rotation ±10° | p=0.4 | p=0.4 | p=0.3 | p=0.3 |
| Brightness/Contrast | p=0.5 | p=0.5 | p=0.3 | p=0.3 |
| GaussianBlur | p=0.3 | p=0.3 | p=0.5 | p=0.5 |
| GaussianNoise | p=0.3 | p=0.3 | p=0.3 | p=0.3 |

### 4-2. get_transform 구현

```python
from torchvision.transforms import v2
import torch

def get_transform(step: int, img_size: int = 512) -> v2.Compose:
    common_post = [
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std =[0.229, 0.224, 0.225]),
    ]

    if step == 1:   # SmartDoc
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10),
            v2.ColorJitter(brightness=0.3, contrast=0.3),
            v2.RandomGrayscale(p=0.5),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            *common_post,
        ])

    elif step == 2: # MIDV-2020
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10),
            v2.ColorJitter(brightness=0.3, contrast=0.3),
            v2.RandomGrayscale(p=0.5),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.RandomPhotometricDistort(p=0.3),
            *common_post,
        ])

    elif step == 3: # PMD 실제
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10),
            v2.Grayscale(num_output_channels=3),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.GaussianBlur(kernel_size=5, sigma=(1.0, 3.0)),
            v2.GaussianNoise(mean=0.0, sigma=0.02),
            *common_post,
        ])

    elif step == 4: # 합성 PMD
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10),
            v2.Grayscale(num_output_channels=3),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.GaussianBlur(kernel_size=5, sigma=(1.0, 3.0)),
            v2.GaussianNoise(mean=0.0, sigma=0.02),
            *common_post,
        ])

    else:
        raise ValueError(f"step은 1~4 사이여야 합니다. 입력값: {step}")
```

### 4-3. 좌표 동시 변환

`v2.RandomHorizontalFlip` / `v2.RandomRotation`은 `tv_tensors.Keypoints`를 함께 전달하면 좌표를 자동 변환합니다.

```python
from torchvision.tv_tensors import Keypoints

kpts  = Keypoints(coords.view(4, 2), canvas_size=(H, W))
img_t, kpts_t = transform(img, kpts)
coords_out = kpts_t.reshape(-1)   # (8,)
```

### 4-4. Stripe Overlay (별도 구현)

`v2.Compose` 호출 전 Dataset의 `__getitem__`에서 적용합니다.

```python
import math, random
import numpy as np

# 수직 fringe 예시
f   = random.uniform(1/30, 1/8)
phi = random.uniform(0, 2 * math.pi)
xs  = np.arange(W, dtype=np.float32)
stripe = A + B * np.cos(2 * math.pi * f * xs + phi)
stripe = np.tile(stripe[np.newaxis, :], (H, 1))

# alpha blending
out = (1 - alpha) * img + alpha * stripe
```

---

## 5. 학습 전략 (4-Step Curriculum)

### 5-1. 전체 흐름

```
Step 1 : SmartDoc-2015 학습
              ↓  checkpoint_step1.pth  /  점검 A
Step 2 : MIDV-2020 학습  (Step 1 모델에서 이어서)
              ↓  checkpoint_step2.pth  /  점검 B
Step 3 : PMD 실제 이미지 Fine-tuning  (Step 2 모델에서 이어서)
              ↓  checkpoint_step3.pth  /  점검 C
Step 4 : 합성 PMD Fringe 재학습  (Step 3 모델에서 이어서)
              ↓  checkpoint_step4.pth  /  점검 D
         최종 모델 확정
```

### 5-2. 단계별 학습 설정

| 단계 | 데이터 | 이미지 수 | Backbone lr | Head lr | Patience |
|---|---|---|---|---|---|
| Step 1 | SmartDoc-2015 | ~25,000장 | 1e-4 | 1e-3 | 15 |
| Step 2 | MIDV-2020 | ~72,000장 | 5e-5 | 5e-4 | 15 |
| Step 3 | PMD 실제 | 50~100장 | 1e-5 | 1e-4 | 10 |
| Step 4 | 합성 PMD | 5,000~10,000장 | 1e-5 | 1e-4 | 15 |

### 5-3. 2단계 Freeze 전략 (각 Step 공통)

```
1단계 (초반 5~10 epoch)
  Backbone : freeze
  Head     : lr 기준값으로 학습
  → Head를 먼저 빠르게 수렴

2단계 (이후 epoch)
  Backbone : unfreeze, lr × 0.1
  Head     : lr 기준값 유지
  → 전체 fine-tuning
```

### 5-4. 옵티마이저 / 스케줄러

```python
optimizer = torch.optim.AdamW(
    [
        {"params": backbone.parameters(), "lr": backbone_lr},
        {"params": head.parameters(),     "lr": head_lr},
    ],
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)
```

### 5-5. Early Stopping

```
val Polygon IoU 기준
patience epoch 동안 개선 없으면 중단
개선 조건 : val_IoU > best_IoU + 1e-4
```

---

## 6. Metric

### 6-1. 공통 함수

```python
import torch
import numpy as np
from shapely.geometry import Polygon


def nme(poly1: torch.Tensor, poly2: torch.Tensor) -> torch.Tensor:
    """Normalized Mean Error (배치 평균)"""
    dist = torch.norm(poly1.view(-1, 4, 2) - poly2.view(-1, 4, 2), dim=2)
    return dist.mean()


def mde(poly1: torch.Tensor, poly2: torch.Tensor,
        img_w: int, img_h: int) -> torch.Tensor:
    """Mean Distance Error (픽셀 단위)"""
    scale = torch.tensor([img_w, img_h] * 4,
                         dtype=poly1.dtype, device=poly1.device)
    dist  = torch.norm(
        (poly1 * scale).view(-1, 4, 2) - (poly2 * scale).view(-1, 4, 2), dim=2
    )
    return dist.mean()


def point_acc(poly1: torch.Tensor, poly2: torch.Tensor) -> dict:
    """Point Accuracy @ 1% / 2% / 5%"""
    dist = torch.norm(poly1.view(-1, 4, 2) - poly2.view(-1, 4, 2), dim=2)
    return {
        "acc1p": (dist < 0.01).float().mean(),
        "acc2p": (dist < 0.02).float().mean(),
        "acc5p": (dist < 0.05).float().mean(),
    }


def polygon_iou(poly1: torch.Tensor, poly2: torch.Tensor) -> torch.Tensor:
    """Polygon IoU (shapely 기반, CPU 연산)"""
    p1 = poly1.detach().cpu().numpy().reshape(-1, 4, 2)
    p2 = poly2.detach().cpu().numpy().reshape(-1, 4, 2)
    ious = []
    for pts1, pts2 in zip(p1, p2):
        try:
            pg1 = Polygon(pts1).buffer(0)
            pg2 = Polygon(pts2).buffer(0)
            inter = pg1.intersection(pg2).area
            union = pg1.union(pg2).area
            ious.append(inter / union if union > 1e-8 else 0.0)
        except Exception:
            ious.append(0.0)
    return torch.tensor(np.mean(ious), dtype=torch.float32)
```

### 6-2. Point Accuracy Threshold 의미

| Threshold | 1000px 기준 픽셀 | 의미 |
|---|---|---|
| @1% (0.01) | ~10px | 표준 기준 / Early Stopping 주 지표 |
| @2% (0.02) | ~20px | 실용 기준 |
| @5% (0.05) | ~50px | 최소 기준 / 초기 수렴 확인 |

### 6-3. SSIM / PSNR (Step 3, 4 전용)

Perspective Transform 결과 이미지 품질을 직접 평가합니다.

```python
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.transform import ProjectiveTransform, warp


def compute_warp_quality(
    image_path: str,
    pred_pts:   np.ndarray,
    gt_pts:     np.ndarray,
    out_size:   tuple = (512, 512),
) -> dict:
    assert os.path.isfile(image_path), f"파일 없음: {image_path}"

    image  = Image.open(image_path).convert("L")
    img_np = np.array(image, dtype=np.float64)
    H, W   = img_np.shape

    scale    = np.array([W, H] * 4, dtype=np.float64)
    src_pred = (pred_pts * scale).reshape(4, 2)
    src_gt   = (gt_pts   * scale).reshape(4, 2)

    ow, oh = out_size
    dst = np.array([[0,0],[ow-1,0],[ow-1,oh-1],[0,oh-1]], dtype=np.float64)

    def _warp(src_pts):
        tf = ProjectiveTransform()
        tf.estimate(dst, src_pts)
        return warp(img_np, tf, output_shape=(oh, ow),
                    order=1, mode="constant", cval=0,
                    preserve_range=True).astype(np.float64)

    warped_pred = _warp(src_pred)
    warped_gt   = _warp(src_gt)

    return {
        "ssim": float(ssim_fn(warped_gt, warped_pred, data_range=255.0)),
        "psnr": float(psnr_fn(warped_gt, warped_pred, data_range=255.0)),
    }
```

### 6-4. Metric 계산 시점

| Metric | Train batch | Val epoch 말 | 단계 완료 후 |
|---|---|---|---|
| SmoothL1 Loss | ✅ | ✅ | - |
| NME | ✅ (GPU) | ✅ | ✅ |
| Polygon IoU | ❌ | ✅ (CPU) | ✅ |
| acc1p / acc2p / acc5p | ❌ | ✅ | ✅ |
| MDE (px) | ❌ | ✅ | ✅ |
| SSIM / PSNR | ❌ | ❌ | ✅ (Step 3, 4) |

---

## 7. 단계별 성능 점검

### 점검 A — Step 1 완료 후 (SmartDoc)

```
측정 대상 : SmartDoc val set

합격 기준:
  Polygon IoU  ≥ 0.90
  NME          ≤ 0.02
  acc1p        ≥ 0.80
  acc2p        ≥ 0.92
  acc5p        ≥ 0.99

→ 합격 : Step 2 진행
→ 불합격 : lr / epoch / augmentation 조정 후 재학습
```

### 점검 B — Step 2 완료 후 (MIDV-2020)

```
측정 대상 1 : MIDV-2020 val set     (새 도메인 적응 확인)
측정 대상 2 : SmartDoc val set      (Forgetting 확인)

합격 기준:
  MIDV val IoU          ≥ 0.90
  SmartDoc IoU 하락     ≤ 0.05

→ 합격 : Step 3 진행
→ Forgetting 발생 : lr 낮추고 Step 2 재학습
```

### 점검 C — Step 3 완료 후 (PMD 실제)

```
측정 대상 1 : PMD val set           (핵심 지표)
측정 대상 2 : SmartDoc + MIDV val   (Forgetting 확인)
측정 대상 3 : SSIM / PSNR          (변환 품질)

합격 기준:
  PMD val IoU           ≥ 0.88
  PMD acc1p             ≥ 0.85
  SSIM                  ≥ 0.90
  PSNR                  ≥ 30dB
  이전 도메인 IoU 하락  ≤ 0.05

→ 합격 : Step 4 진행
→ 불합격 : PMD 데이터 추가 수집 또는 lr 조정 후 재학습
```

### 점검 D — Step 4 완료 후 (최종)

```
측정 대상 : 전체 4개 도메인 val set

최종 합격 기준:
  PMD val IoU           ≥ 0.90
  PMD acc1p             ≥ 0.90
  SSIM                  ≥ 0.92
  PSNR                  ≥ 32dB
  전 도메인 IoU 차      ≤ 0.05

→ 합격 : 최종 모델 확정 (best_model.pth)
→ 불합격 : 합성 데이터 조건 보강 후 Step 4 재학습
```

### 7-1. StepMetric 클래스

```python
class StepMetric:
    FORGETTING_THRESHOLD = 0.05
    PASS_CRITERIA = {
        1: {"iou":0.90, "nme":0.020, "acc1p":0.80, "acc2p":0.92, "acc5p":0.99},
        2: {"iou":0.90, "nme":0.020, "acc1p":0.80, "acc2p":0.92, "acc5p":0.99},
        3: {"iou":0.88, "nme":0.015, "acc1p":0.85, "acc2p":0.95, "acc5p":0.99},
        4: {"iou":0.90, "nme":0.015, "acc1p":0.90, "acc2p":0.97, "acc5p":0.99},
    }

    def __init__(self, step: int, img_w: int = 1920, img_h: int = 1080,
                 prev: dict = None):
        self.step  = step
        self.img_w = img_w
        self.img_h = img_h
        self.prev  = prev or {}

    def compute(self, poly_pred, poly_gt) -> dict:
        acc = point_acc(poly_pred, poly_gt)
        return {
            "iou"  : polygon_iou(poly_pred, poly_gt),
            "nme"  : nme(poly_pred, poly_gt),
            "mde"  : mde(poly_pred, poly_gt, self.img_w, self.img_h),
            "acc1p": acc["acc1p"],
            "acc2p": acc["acc2p"],
            "acc5p": acc["acc5p"],
        }

    def check_pass(self, result: dict) -> dict:
        criteria  = self.PASS_CRITERIA[self.step]
        item_pass = {
            "iou"  : result["iou"].item()   >= criteria["iou"],
            "nme"  : result["nme"].item()   <= criteria["nme"],
            "acc1p": result["acc1p"].item() >= criteria["acc1p"],
            "acc2p": result["acc2p"].item() >= criteria["acc2p"],
            "acc5p": result["acc5p"].item() >= criteria["acc5p"],
        }
        forgetting = {}
        if self.step >= 2 and self.prev:
            current_iou = result["iou"].item()
            for domain, prev_iou in self.prev.items():
                drop = prev_iou - current_iou
                forgetting[domain] = {
                    "prev_iou": prev_iou, "current_iou": current_iou,
                    "drop": drop, "pass": drop <= self.FORGETTING_THRESHOLD,
                }
        return {
            "pass"      : all(item_pass.values()) and
                          all(v["pass"] for v in forgetting.values()),
            "criteria"  : item_pass,
            "forgetting": forgetting,
        }

    def report(self, result: dict, domain: str = "") -> None:
        check  = self.check_pass(result)
        prefix = f"[Step {self.step}]" + (f" {domain}" if domain else "")
        status = "PASS ✓" if check["pass"] else "FAIL ✗"
        print(f"\n{prefix}  {status}")
        print(f"  IoU   : {result['iou'].item():.4f}  "
              f"(>= {self.PASS_CRITERIA[self.step]['iou']})"
              f"  {'✓' if check['criteria']['iou']   else '✗'}")
        print(f"  NME   : {result['nme'].item():.4f}  "
              f"(<= {self.PASS_CRITERIA[self.step]['nme']})"
              f"  {'✓' if check['criteria']['nme']   else '✗'}")
        print(f"  MDE   : {result['mde'].item():.2f} px")
        print(f"  acc1p : {result['acc1p'].item():.4f}  "
              f"(>= {self.PASS_CRITERIA[self.step]['acc1p']})"
              f"  {'✓' if check['criteria']['acc1p'] else '✗'}")
        print(f"  acc2p : {result['acc2p'].item():.4f}  "
              f"(>= {self.PASS_CRITERIA[self.step]['acc2p']})"
              f"  {'✓' if check['criteria']['acc2p'] else '✗'}")
        print(f"  acc5p : {result['acc5p'].item():.4f}  "
              f"(>= {self.PASS_CRITERIA[self.step]['acc5p']})"
              f"  {'✓' if check['criteria']['acc5p'] else '✗'}")
        if check["forgetting"]:
            print("  --- Forgetting ---")
            for d, v in check["forgetting"].items():
                print(f"  {d:12s} : {v['prev_iou']:.4f} → "
                      f"{v['current_iou']:.4f}  drop={v['drop']:.4f}  "
                      f"{'✓' if v['pass'] else '✗'}")
```

---

## 8. Inference 파이프라인

### 8-1. 입력 전처리 (GaussianBlur)

fringe 줄무늬(고주파)를 제거해 학습 도메인과 유사하게 만듭니다.

```python
from PIL import Image, ImageFilter
from torchvision.transforms import v2
import torch

def preprocess_pmd(image_path: str, img_size: int = 512,
                   blur_sigma: float = 2.0) -> torch.Tensor:
    img  = Image.open(image_path).convert("L")
    img  = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    img3 = Image.merge("RGB", [img, img, img])
    t    = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std =[0.229, 0.224, 0.225]),
    ])(img3)
    return t.unsqueeze(0)   # (1, 3, H, W)
```

### 8-2. 추론 및 좌표 반환

```python
model.eval()
with torch.no_grad():
    img_t  = preprocess_pmd("fringe.png")
    coords = model(img_t).squeeze(0)   # (8,) sigmoid 출력 [0,1]

x1,y1, x2,y2, x3,y3, x4,y4 = coords.tolist()
```

### 8-3. Perspective Transform 적용

```python
import numpy as np
from skimage.transform import ProjectiveTransform, warp
from PIL import Image

image  = Image.open("fringe.png").convert("L")
img_np = np.array(image)
H, W   = img_np.shape

scale   = np.array([W, H] * 4)
src_pts = (coords.numpy() * scale).reshape(4, 2)

ow, oh = 1024, 1024
dst = np.array([[0,0],[ow-1,0],[ow-1,oh-1],[0,oh-1]], dtype=np.float64)

tf = ProjectiveTransform()
tf.estimate(dst, src_pts)
warped = warp(img_np, tf, output_shape=(oh, ow),
              order=1, preserve_range=True)
```

---

## 9. 체크포인트 관리

```
checkpoints/
├── checkpoint_step1.pth   ← Step 1 best model (SmartDoc)
├── checkpoint_step2.pth   ← Step 2 best model (MIDV-2020)
├── checkpoint_step3.pth   ← Step 3 best model (PMD 실제)
└── checkpoint_step4.pth   ← Step 4 best model (최종)
```

각 단계 시작 시 이전 단계 best model을 로드합니다.

```python
import os
import torch

model.load_state_dict(torch.load(
    os.path.join("checkpoints", f"checkpoint_step{step-1}.pth")
))
```

---

## 10. 전체 파이프라인 요약

```
[데이터 준비]
SmartDoc / MIDV-2020 다운로드 → 공통 CSV 변환
PMD 촬영 → LabelMe 라벨링 → CSV 저장
합성 PMD fringe 생성 → CSV 자동 저장

[Step 1] SmartDoc 학습
  get_transform(step=1) + StripeOverlay(p=0.4)
  Backbone lr=1e-4, Head lr=1e-3
  Early Stop: val IoU patience=15
  → 점검 A: IoU≥0.90, acc1p≥0.80

[Step 2] MIDV-2020 학습 (Step 1 이어서)
  get_transform(step=2) + StripeOverlay(p=0.5)
  Backbone lr=5e-5, Head lr=5e-4
  → 점검 B: IoU≥0.90, Forgetting≤0.05

[Step 3] PMD Fine-tuning (Step 2 이어서)
  get_transform(step=3)
  Backbone lr=1e-5, Head lr=1e-4
  → 점검 C: IoU≥0.88, SSIM≥0.90, PSNR≥30dB

[Step 4] 합성 PMD 재학습 (Step 3 이어서)
  get_transform(step=4)
  Backbone lr=1e-5, Head lr=1e-4
  → 점검 D: IoU≥0.90, SSIM≥0.92, PSNR≥32dB

[Inference]
PMD fringe 이미지
  → GaussianBlur(σ=2.0)
  → 3채널 복제 + ImageNet 정규화
  → 모델 추론 (Sigmoid)
  → 4점 좌표 [x1,y1, x2,y2, x3,y3, x4,y4]
  → Perspective Transform
  → 보정된 OLED 패널 이미지
```