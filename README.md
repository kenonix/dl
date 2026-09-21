# 🦾 EMG ML Pipeline (`emg_ml_pipeline`)

> **Rust & Burn 프레임워크 기반의 고성능 실시간 근전도(EMG) 신호 처리 및 딥러닝 분류 파이프라인**  
> 5채널 아날로그 센서 데이터로부터 실시간 FFT 주파수 스펙트럼 및 Hudgins 4대 시간 도메인 특징(총 65차원)을 추출하고, 1D-TCN(Temporal Convolutional Network)을 비롯한 딥러닝 모델을 통해 실시간 손동작 및 제스처를 초저지연·고정밀로 판정합니다.

---

## 📌 목차
- [1. 프로젝트 개요](#1-프로젝트-개요)
- [2. 🧠 적용된 핵심 AI / 딥러닝 / 신호처리 기법 총정리](#2--적용된-핵심-ai--딥러닝--신호처리-기법-총정리)
  - [2.1 생체 신호 처리 및 특징 공학 (Biomedical Feature Engineering)](#21-생체-신호-처리-및-특징-공학-biomedical-feature-engineering)
  - [2.2 데이터 증강 및 정규화 (Data Augmentation & Normalization)](#22-데이터-증강-및-정규화-data-augmentation--normalization)
  - [2.3 딥러닝 신경망 아키텍처 (Deep Neural Architectures)](#23-딥러닝-신경망-아키텍처-deep-neural-architectures)
  - [2.4 학습 최적화 및 동역학 (Optimization & Training Dynamics)](#24-학습-최적화-및-동역학-optimization--training-dynamics)
  - [2.5 실시간 추론 안정화 및 후처리 (Inference Stabilization & Post-Processing)](#25-실시간-추론-안정화-및-후처리-inference-stabilization--post-processing)
- [3. 특징 추출 파이프라인 (65차원 DC-Invariant AC 특징)](#3-특징-추출-파이프라인-65차원-dc-invariant-ac-특징)
- [4. 지원 딥러닝 모델 아키텍처](#4-지원-딥러닝-모델-아키텍처)
- [5. 실시간 추론 안정화 및 노이즈 극복 기법](#5-실시간-추론-안정화-및-노이즈-극복-기법)
- [6. 시스템 구성 및 아키텍처](#6-시스템-구성-및-아키텍처)
- [7. 실행 및 사용 가이드](#7-실행-및-사용-가이드)
- [8. 파일 및 디렉토리 구조](#8-파일-및-디렉토리-구조)
- [9. 제스처 클래스 정의](#9-제스처-클래스-정의-num_classes-4)

---

## 1. 프로젝트 개요

본 프로젝트는 근전도 센서(MyoWare 등 5채널 아날로그 센서)에서 발생하는 미세 전기 신호를 수집, 정규화, 학습 및 실시간 추론하는 엔드투엔드 시스템입니다.
- **클라이언트 (`emg_client`)**: 시리얼 통신을 통한 데이터 수집, 로컬/원격 학습 요청, 실시간 추론 CLI
- **원격 학습 서버 (`emg_server`)**: 고성능 GPU(WGPU) 가속을 활용하여 대규모 데이터셋 원격 학습 및 모델 제공 REST API
- **프레임워크**: 순수 Rust 기반 머신러닝 엔진 [Burn](https://burn.dev/) (WGPU 백엔드)

---

## 2. 🧠 적용된 핵심 AI / 딥러닝 / 신호처리 기법 총정리

본 파이프라인에 적용된 모든 인공지능, 통계적 머신러닝, 생체 신호 처리 기법의 이론적 배경과 구현 내역입니다.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            🧠 종합 AI & 신호 처리 파이프라인                              │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ [1. 신호 입력] 5ch 아날로그 EMG (50Hz)                                                   │
│        │                                                                                 │
│        ▼                                                                                 │
│ [2. 특징 공학] 윈도우(N=16) Detrending ➔ 65차원 DC-Invariant 벡터 산출                    │
│                • 국소 중심화 신호 (5ch)                                                  │
│                • Detrended FFT 크기 스펙트럼 (40ch)                                      │
│                • Hudgins AC 시간 지표 (AC-RMS, AC-MAV, WL, ZC) (20ch)                   │
│        │                                                                                 │
│        ▼                                                                                 │
│ [3. 데이터 증강] (학습 단계) 3배 확장 파이프라인                                         │
│                • Random Baseline Shift (±50 ADC count)                                   │
│                • Gaussian Jittering (σ=2.5)                                              │
│                • Amplitude Scaling (0.85 ~ 1.20×)                                        │
│        │                                                                                 │
│        ▼                                                                                 │
│ [4. 정규화] Feature-wise Z-Score Standardization (x - μ) / σ                             │
│        │                                                                                 │
│        ▼                                                                                 │
│ [5. 신경망 모델] 1D-TCN (Temporal Convolutional Network)                                 │
│                • Dilated Conv1d (Dilation: 1, 2, 4 지수적 Receptive Field 확장)          │
│                • Batch Normalization (BatchNorm1d)                                       │
│                • Residual Shortcut Connection (x + f(x))                                 │
│                • Spatial/Temporal Dropout (p=0.15)                                       │
│                • Global Adaptive Average Pooling                                         │
│        │                                                                                 │
│        ▼                                                                                 │
│ [6. 추론 후처리] 실시간 안정화 필터링                                                    │
│                • 1.5초 휴식기 영점 자동 보정 (Rest Baseline Zeroing)                     │
│                • 생체 신호 에너지 기반 스마트 노이즈 게이트 (Total AC-RMS < 75 ➔ Rest)   │
│                • 9-프레임 다수결 스무딩 필터 (Majority Voting Buffer)                    │
│                • 수치적 안정 Softmax 신뢰도(Confidence) & 안정도(Stability) 지수         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 생체 신호 처리 및 특징 공학 (Biomedical Feature Engineering)

| 기법명 | 수학적 정의 / 원리 | 적용 목적 및 효과 |
| :--- | :--- | :--- |
| **시계열 슬라이딩 윈도우**<br>(Sliding Window) | 윈도우 크기 $N=30$ (약 0.6초 분량) | 과거 노이즈 간섭을 최소화하면서도 제스처의 동적 시간 전이 패턴을 온전히 포착 (반응 속도 2배 향상) |
| **국소 중심화 변위**<br>(Centered Raw Displacement) | $x_{\text{centered}}[i] = x[i] - \mu_{\text{window}}$ | 센서 착용 시마다 달라지는 직류 기준 전압(34~604)을 제거하고 순간 교류 변위만 5차원으로 산출 |
| **Detrended 16-FFT**<br>(비직류 고속 푸리에 변환) | $X[k] = \sum_{n=0}^{N-1} (x[n] - \mu) e^{-j\frac{2\pi}{N}kn}$<br>Magnitude: $\sqrt{\text{Re}^2 + \text{Im}^2}$ | 윈도우 평균을 빼는 Detrending을 통해 저주파~고주파 전 대역으로의 직류 누설(Spectral Leakage)을 차단하고 채널당 8개 주파수 대역 에너지 산출 |
| **AC-RMS (국소 표준편차)**<br>(Root Mean Square) | $\text{AC-RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2}$ | 직류 전압을 제외한 순수 근육 수축 전력(AC Power)만을 추출. 센서를 다시 착용해도 휴식/수축 구분이 완벽히 유지됨 |
| **AC-MAV (국소 평균절대편차)**<br>(Mean Absolute Value) | $\text{AC-MAV} = \frac{1}{N}\sum_{i=1}^N \|x_i - \mu\|$ | 근육 수축 신호의 순수 진폭 포락선(Envelope)을 산출 |
| **파형 길이 (WL)**<br>(Waveform Length) | $\text{WL} = \sum_{i=1}^{N-1} \|x_i - x_{i-1}\|$ | 차분 신호 기반으로 신호의 누적 변동량과 복잡도를 표현 (본질적으로 직류 불변) |
| **영점 교차율 (ZC)**<br>(Zero Crossing Rate) | $x_i - \mu$의 부호 변경 횟수<br>(임계치 $\|x_i - x_{i-1}\| \ge 3.0$ ADC) | 주파수 밀도를 간접 반영하며, 노이즈 불감대(Deadzone)를 두어 미세 아날로그 전자파 잡음 필터링 |
| **완전 DC-Invariant 공간**<br>(DC-Invariant Feature Space) | 총 65차원 전체가 교류 성분으로만 구성 | 센서 위치 이동, 전극 재착용, 피부 임피던스 변화에도 불구하고 특징 공간의 왜곡이 0에 수렴 |

---

### 2.2 데이터 증강 및 정규화 (Data Augmentation & Normalization)

1. **무작위 베이스라인 시프트 (Random Baseline Shift)**:
   - 채널별로 $\pm 50$ ADC 단위의 임의 오프셋 $\Delta \in [-50, 50]$을 신호에 주입하여 학습.
   - 모델이 전극 재착용으로 인한 기준선 표류(Baseline Drift)를 제스처 변화로 착각하지 않도록 적응성 부여.
2. **가우시안 노이즈 지터링 (Gaussian Noise Jittering)**:
   - $\mathcal{N}(0, \sigma^2)$ ($\sigma=2.5$) 잡음을 신호에 합성하여 피부 접촉 불량이나 아날로그 전원 노이즈에 대한 내성 훈련.
3. **진폭 스케일링 (Amplitude Scaling)**:
   - 신호에 $0.85 \sim 1.20\times$ 임의 스케일을 곱하여, 사용자의 악력 차이나 근육 피로도(Fatigue)로 인한 수축 강도 약화 상황에서도 동일 제스처로 정확히 인식하도록 일반화.
4. **특징별 Z-Score 표준화 (Feature-wise Z-Score Standardization)**:
   - $x_{\text{norm}} = \frac{x - \mu}{\sigma + 10^{-4}}$ 공식을 통해 65개 모든 특징 열을 평균 0, 표준편차 1로 스케일링하여 특정 채널의 가중치 독점 방지.
5. **학습-추론 분포 동기화 (`NormalizationStats`)**:
   - 학습 완료 시 산출된 평균/표준편차 벡터를 JSON으로 보존하고 실시간 추론 시 100% 동일하게 로드하여 공변량 불일치(Covariate Shift) 차단.

---

### 2.3 딥러닝 신경망 아키텍처 (Deep Neural Architectures)

#### 🥇 1D-TCN (Temporal Convolutional Network, 메인 모델)
시계열 데이터에서 RNN/LSTM의 순차 처리 한계를 극복하고 병렬 처리와 넓은 수용 영역을 동시에 달성한 최신 시계열 아키텍처입니다.

```
[입력 텐서: Batch × 65 Features × 30 Seq]
               │
               ▼
   ┌───────────────────────┐
   │ TcnBlock 1 (d=1, c=32)│ ──> Conv1d(k=3, pad=1) ➔ BatchNorm ➔ ReLU ➔ Dropout(0.15) ➔ Conv1d ➔ BatchNorm ➔ Dropout ➔ (+) Residual
   └───────────────────────┘
               │
               ▼
   ┌───────────────────────┐
   │ TcnBlock 2 (d=2, c=64)│ ──> Conv1d(k=3, pad=2, d=2) ➔ BatchNorm ➔ ReLU ➔ Dropout ➔ Conv1d ➔ BatchNorm ➔ (+) 1x1 Conv Shortcut
   └───────────────────────┘
               │
               ▼
   ┌───────────────────────┐
   │ TcnBlock 3 (d=4, c=64)│ ──> Conv1d(k=3, pad=4, d=4) ➔ BatchNorm ➔ ReLU ➔ Dropout ➔ Conv1d ➔ BatchNorm ➔ (+) Residual
   └───────────────────────┘
               │
               ▼
   [AdaptiveAvgPool1d(1)]   ──> 시계열 전체를 [Batch × 64]로 글로벌 평균 풀링
               │
               ▼
   [Linear(64 ➔ 32) + ReLU]
               │
               ▼
   [Dropout(0.15)]
               │
               ▼
   [Linear(32 ➔ 4)]         ──> 4개 클래스 최종 로짓(Logits)
```

- **Dilated Convolutions (확장 합성곱)**: Dilation 계수를 $1, 2, 4$로 기하급수적으로 확장하여, 적은 파라미터로 30개 시계열 전체(Receptive Field $\ge 30$)를 한 번에 조망.
- **Residual Connections (잔여 연결)**: 입력을 출력에 직접 더하는 Shortcut($x + \mathcal{F}(x)$)을 배치하여 깊은 네트워크에서도 그래디언트 소실(Gradient Vanishing)을 원천 방지.
- **Batch Normalization (배치 정규화)**: 각 블록마다 활성화 값의 평균과 분산을 재정규화하여 내부 공변량 변화(Internal Covariate Shift)를 제거하고 초고속 수렴 보장.
- **Dropout Regularization (드롭아웃, $p=0.15$)**: 뉴런의 무작위 비활성화를 통해 특정 특징 간의 취약한 상호 적응(Co-adaptation)을 차단하고 노이즈 강건성 극대화.
- **Global Adaptive Pooling**: 시계열 길이에 무관하게 견고한 글로벌 표현 벡터를 추출하고 FC 파라미터 수를 대폭 절감.

#### 기타 지원 모델
- **1D-CNN**: 2개 합성곱 블록 + `BatchNorm1d` 기반 초경량 국소 특징 추출기.
- **CRNN (CNN + LSTM)**: Conv1d 공간 특징 추출 후 LSTM 순환 계층을 통과하여 시계열 전이를 학습하는 하이브리드 모델.
- **Pure LSTM**: 고전 순환 신경망 모델.

---

### 2.4 학습 최적화 및 동역학 (Optimization & Training Dynamics)

- **Adam 옵티마이저 (Adaptive Moment Estimation)**:
  - 1차 모멘트($m_t$, 그래디언트 지수 이동 평균)와 2차 모멘트($v_t$, 그래디언트 제곱 이동 평균)를 조합하여 가중치별 적응형 학습률 적용.
- **학습률 최적화 (Learning Rate Tuning)**:
  - 기존 $0.01$ (손실 고원 현상 및 가중치 발산 원인) ➔ **$0.001$ ($1\times 10^{-3}$)**로 안정화하여 손실이 `1.28` ➔ `0.04`로 부드럽게 수렴.
- **Cross-Entropy Loss**:
  - $\mathcal{L} = -\sum_{c=1}^C y_c \log(p_c)$ 기반의 다중 클래스 분류 손실 함수.
- **WGPU 텐서 병렬화**:
  - Rust Burn 엔진을 기반으로 GPU 컴퓨트 셰이더를 구동하여 행렬 연산 가속.

---

### 2.5 실시간 추론 안정화 및 후처리 (Inference Stabilization & Post-Processing)

- **1.5초 휴식기 영점 자동 보정 (Rest Baseline Zeroing)**:
  - 센서 부착 직후 사용자의 개별 피부 저항 및 장착 위치에 따른 직류 기준선을 75개 샘플로 실시간 측정하여 정밀 차감.
- **스마트 에너지 노이즈 게이트 (Rest Protection Noise Gate)**:
  - 근전도 생체 신호의 물리적 특성상, 팔이 쉴 때의 5채널 총 교류 수축 에너지($\sum \text{AC-RMS}$)는 평균 `61.4` (최대 75 미만), 실제 제스처 수축 시는 `117 ~ 166`으로 명확히 양분됩니다.
  - 총 에너지가 75 미만일 경우 모델의 출력을 강제로 `0: 휴식 (Relax)`으로 고정하여, 가만히 쉬고 있을 때의 오작동을 수학적으로 원천 차단.
- **다수결 스무딩 필터 (Majority Voting Buffer, Window: 9)**:
  - $\hat{y} = \text{mode}(y_{t-8}, y_{t-7}, \dots, y_t)$
  - 최근 9프레임(약 0.18초)의 추론 결과를 모아 최빈값을 최종 제스처로 확정함으로써 0.02초 단위 순간 떨림/깜빡임(Jitter)을 제거.
- **수치적 안정 Softmax 및 신뢰도/안정도 지표**:
  - $p_i = \frac{\exp(z_i - \max(z))}{\sum_j \exp(z_j - \max(z))}$ 오버플로우 방지 Softmax 확률 계산.
  - 모델의 순수 확신도(Confidence %)와 다수결 큐의 일치율(Stability: $K/9$)을 실시간 터미널에 표시.

---

## 3. 특징 추출 파이프라인 (65차원 DC-Invariant AC 특징)

신호의 단순 원시값에 의존하지 않고, 근육 수축 상태와 주파수 특성을 직류(DC) 편향 없이 추출하는 **65차원 완전 직류 불변(DC-Invariant) 특징 벡터**를 실시간 산출합니다. 센서를 재착용하여 기본 전압이 바뀌어도 값이 왜곡되지 않습니다.

$$\text{Total Features (65)} = \text{Centered Raw (5)} + \text{Detrended FFT (40)} + \text{Hudgins AC Time-Domain (20)}$$

```
[5채널 아날로그 신호] ──┬──> [1] 국소 중심화 신호 (raw - mean) (5차원)
                       ├──> [2] 16-point Detrended FFT Magnitude (채널당 8 bin × 5 = 40차원)
                       └──> [3] Hudgins AC 시간 도메인 지표 (채널당 4개 × 5 = 20차원)
                                 ├─ AC-RMS (국소 표준편차): 순수 교류 근육 수축 총 파워
                                 ├─ AC-MAV (국소 평균절대편차): 순수 진폭 포락선
                                 ├─ WL (Waveform Length): 신호 누적 변동량 (차분 지표)
                                 └─ ZC (Zero Crossing): 평균선 교차 횟수 (주파수 대용)
```

---

## 4. 지원 딥러닝 모델 아키텍처

| 모델명 | 파일 위치 | 특징 및 구조 |
| :--- | :--- | :--- |
| **1D-TCN** (🥇 추천) | [`src/models/tcn.rs`](src/models/tcn.rs) | **Dilated Conv1d(Dilation: 1, 2, 4) + Residual Connection + BatchNorm1d + Dropout(0.15)**<br>파라미터 폭증 없이 넓은 수용 영역을 확보하고 드롭아웃으로 노이즈 과적합을 방지하여 강건성 극대화 |
| **1D-CNN** | [`src/models/cnn.rs`](src/models/cnn.rs) | Conv1d + BatchNorm1d + AdaptiveAvgPool1d 기반 경량 합성곱 모델 |
| **CNN + LSTM** | [`src/models/cnn_lstm.rs`](src/models/cnn_lstm.rs) | 국소적 특징 추출(Conv+BN) 후 순차적 시계열 패턴 학습(LSTM) 결합 |
| **LSTM** | [`src/models/lstm.rs`](src/models/lstm.rs) | 순수 순환 신경망 구조 |

---

## 5. 실시간 추론 안정화 및 노이즈 극복 기법

### 1) 1.5초 휴식기 영점 자동 보정 (Rest Baseline Zeroing)
- 실시간 추론(3번 메뉴) 시작 시 1.5초간 팔에 힘을 뺀 상태(75개 샘플)를 자동 측정하여 현재 착용 상태의 기준선을 등록합니다.
- 센서를 다시 착용하거나 위치가 바뀌더라도 즉시 영점이 동기화됩니다.

### 2) 스마트 에너지 노이즈 게이트 (Rest 보호 필터)
- 5개 채널의 총 AC-RMS 에너지(수축 에너지)가 75 미만이면 팔이 휴식 중인 상태로 판정하여 강제로 `0: 휴식 (Relax)` 판정을 고정합니다. 미세한 잡음으로 인해 제스처가 오작동하는 것을 원천 차단합니다.

### 3) 학습 시 3배 데이터 증강 (Data Augmentation)
- 학습 시 채널별 무작위 베이스라인 시프트($\pm 50$), 가우시안 노이즈, 진폭 스케일링($0.85 \sim 1.20$)을 자동 주입하여 다양한 착용 오차와 악력 변화에 둔감한 강건 모델을 훈련합니다.

### 4) 다수결 스무딩 필터 (Majority Voting Buffer, 크기: 9)
- 최근 9개 프레임(약 0.18초)의 추론 결과를 모아 최빈값(다수결)으로 최종 제스처를 결정하여 순간 깜빡임(Jitter)을 방지합니다.

### 5) Softmax 확률 및 신뢰도/안정도/근육 에너지 UI
- 모델 출력 로짓을 확률(0~100%)로 변환하여 확신도(Confidence), 다수결 일치율(Stability), 실시간 근육 수축 에너지를 함께 출력합니다.
  ```text
  🤖 [1D-TCN  ] ➔ 주먹 쥐기 (Fist)   (신뢰도: 98.5%, 안정도: 9/9) | 에너지: 145.2 | 센서: [330, 605, 150, 600, 35]
  ```

---

## 6. 시스템 구성 및 아키텍처

```
┌────────────────────────────────────────┐         ┌───────────────────────────────────────┐
│        emg_client (로컬 장치)          │         │        emg_server (GPU 학습 서버)     │
│                                        │         │                                       │
│  [아두이노/센서] -> 시리얼 수신 (50Hz) │         │  • POST /api/upload (데이터 업로드)   │
│         │                              │  HTTP   │  • POST /api/train  (원격 WGPU 학습)  │
│  [특징 추출 (65D)] -> 슬라이딩 윈도우  ├────────>│  • GET  /api/status (실시간 에포크)   │
│         │                              │<────────┤  • GET  /api/download/:name           │
│  [추론 + 9프레임 다수결 필터]          │         │    (.mpk 가중치 + 정규화 JSON 동시수신)│
└────────────────────────────────────────┘         └───────────────────────────────────────┘
```

---

## 7. 실행 및 사용 가이드

### 빌드 및 사전 준비
```bash
# 디버그 검사
cargo check --bins --tests

# 릴리즈 최적화 빌드
cargo build --release
```

### 1) 원격 학습 서버 구동 (선택)
```bash
cargo run --release --bin emg_server
# 기본 포트: 0.0.0.0:3000
```

### 2) 클라이언트 실행
```bash
cargo run --release --bin emg_client
```

콘솔 대화형 메뉴가 표시됩니다:
```text
========================================
   EMG 머신러닝/딥러닝 통합 파이프라인
========================================
1. 센서 데이터 수집 (CSV 저장)
2. 로컬 모델 학습 (WGPU 가속)
3. 실시간 제스처 추론 (동작 분류)
4. 원격 서버 연동 (서버 학습 & 모델 다운로드)
5. 프로그램 종료
```

- **1번 메뉴**: 5채널 센서 데이터를 라벨별로 수집하여 `emg_dataset_YYYYMMDD_HHMMSS.csv`로 저장.
- **2번 메뉴 / 4번 메뉴**: `[1] 1D-TCN`을 선택하여 150 에포크 학습.
  - 학습 완료 시 `emg_tcn_model.mpk` 및 `norm_stats_emg_tcn_model.json` 동시 생성.
- **3번 메뉴**: 실시간 추론 시작. `✔ [정규화 통계 로드 완료]` 확인 후 센서 움직임에 따라 제스처 판정.

---

## 8. 파일 및 디렉토리 구조

```
dl/
├── Cargo.toml                      # 프로젝트 의존성 및 바이너리 정의
├── norm_stats_emg_tcn_model.json   # 1D-TCN 학습 데이터셋 65차원 정규화 통계
├── emg_tcn_model.mpk               # 훈련 완료된 1D-TCN 모델 바이너리
├── src/
│   ├── lib.rs                      # 코어 라이브러리 인터페이스 정의
│   ├── main.rs                     # emg_client 바이너리 엔트리포인트
│   ├── fft.rs                      # 실시간 16-FFT 및 Hudgins 4대 특징 추출기 (65차원)
│   ├── train.rs                    # 데이터 로딩, Z-Score 표준화, WGPU 모델 학습 루프
│   ├── bin/
│   │   └── server.rs               # Axum 기반 원격 WGPU 학습 서버
│   └── models/
│       ├── mod.rs                  # 모델 아키텍처 열거형 및 인터페이스
│       ├── tcn.rs                  # 1D-TCN (Dilated Conv + Residual + BatchNorm)
│       ├── cnn.rs                  # 1D-CNN (BatchNorm 적용)
│       ├── cnn_lstm.rs             # CRNN (CNN + LSTM + BatchNorm)
│       └── lstm.rs                 # 기본 LSTM 모델
└── tests/
    └── training_integration_test.rs # 모델 학습 및 수렴 검증 통합 테스트
```

---

## 9. 제스처 클래스 정의 (`NUM_CLASSES: 4`)

- `Class 0`: 휴식 상태 (Rest)
- `Class 1`: 주먹 쥐기 (Fist)
- `Class 2`: 손가락 펴기 (Open Hand)
- `Class 3`: 손목 젖히기 (Wrist Flexion)
