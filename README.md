# 🦾 EMG ML Pipeline (`emg_ml_pipeline`)

> **Rust & Burn 프레임워크 기반의 고성능 실시간 근전도(EMG) 신호 처리 및 딥러닝 분류 파이프라인**  
> 5채널 아날로그 센서 데이터로부터 실시간 FFT 주파수 스펙트럼 및 생체 전자기학 기반 확장 시간 도메인 6대 지표(총 75차원 DC-Invariant 특징)를 추출하고, **정적 1D-TCN(Temporal Convolutional Network)** 및 **지능형 추론 안정화 엔진(확률 EMA + 히스테리시스 락 + 디바운싱)**을 통해 센서 탈착/재착용 노이즈와 출력 채터링(Chattering)을 완전히 극복한 초저지연·고정밀 제스처 판정 시스템입니다.

---

## 📌 목차
- [1. 프로젝트 개요](#1-프로젝트-개요)
- [2. 🧠 적용된 핵심 AI / 딥러닝 / 신호처리 기법 총정리](#2--적용된-핵심-ai--딥러닝--신호처리-기법-총정리)
  - [2.1 생체 신호 처리 및 특징 공학 (75차원 DC-Invariant Feature Engineering)](#21-생체-신호-처리-및-특징-공학-75차원-dc-invariant-feature-engineering)
  - [2.2 데이터 증강, 클래스 균형화 및 정규화 (Augmentation, Balancing & Normalization)](#22-데이터-증강-클래스-균형화-및-정규화-augmentation-balancing--normalization)
  - [2.3 딥러닝 신경망 아키텍처 (정적 1D-TCN Architecture)](#23-딥러닝-신경망-아키텍처-정적-1d-tcn-architecture)
  - [2.4 학습 최적화 및 정규화 동역학 (Label Smoothing & Optimization)](#24-학습-최적화-및-정규화-동역학-label-smoothing--optimization)
  - [2.5 지능형 실시간 추론 안정화 & 4대 동작 개별 확률 모니터링 (`GestureStabilizer`)](#25-지능형-실시간-추론-안정화--4대-동작-개별-확률-모니터링-gesturestabilizer)
- [3. 특징 추출 파이프라인 (75차원 완전 직류 불변 공간)](#3-특징-추출-파이프라인-75차원-완전-직류-불변-공간)
- [4. 지원 딥러닝 모델 아키텍처](#4-지원-딥러닝-모델-아키텍처)
- [5. 채터링 제로 & 노이즈 극복 3중 방어 체계](#5-채터링-제로--노이즈-극복-3중-방어-체계)
- [6. 시스템 구성 및 네트워크 통신](#6-시스템-구성-및-네트워크-통신)
- [7. 실행 및 사용 가이드](#7-실행-및-사용-가이드)
- [8. 파일 및 디렉토리 구조](#8-파일-및-디렉토리-구조)
- [9. 제스처 클래스 정의](#9-제스처-클래스-정의-num_classes-4)

---

## 1. 프로젝트 개요

본 프로젝트는 근전도 센서(MyoWare 등 5채널 아날로그 센서)에서 발생하는 미세 전기 신호를 수집, 정규화, 학습 및 실시간 추론하는 엔드투엔드 시스템입니다.
- **클라이언트 (`emg_client`)**: 시리얼 통신을 통한 50Hz 데이터 수집, 로컬/원격 학습 요청, 4대 동작 개별 확률 실시간 모니터링 CLI
- **원격 학습 서버 (`emg_server`)**: 고성능 GPU(WGPU) 가속을 활용하여 대규모 데이터셋 원격 학습 및 모델/정규화 통계 제공 REST API
- **프레임워크**: 순수 Rust 기반 머신러닝 엔진 [Burn](https://burn.dev/) (WGPU 백엔드)

---

## 2. 🧠 적용된 핵심 AI / 딥러닝 / 신호처리 기법 총정리

본 파이프라인에 적용된 모든 인공지능, 통계적 머신러닝, 생체 신호 처리 기법의 이론적 배경과 구현 내역입니다.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            🧠 종합 AI & 신호 처리 파이프라인                              │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ [1. 신호 입력] 5ch 아날로그 EMG (50Hz 샘플링)                                            │
│        │                                                                                 │
│        ▼                                                                                 │
│ [2. 특징 공학] 윈도우(N=16) Detrending ➔ 75차원 DC-Invariant 벡터 산출                    │
│                • 국소 중심화 신호 (5ch)                                                  │
│                • Detrended FFT 크기 스펙트럼 (40ch)                                      │
│                • 확장 시간 도메인 6대 지표 (30ch)                                         │
│                  - AC-RMS, AC-MAV, WL, ZC                                                │
│                  - SSC (기울기 부호 역전율), WAMP (윌리슨 진폭 변동율)                   │
│        │                                                                                 │
│        ▼                                                                                 │
│ [3. 데이터 전처리 & 증강] 1:1:1:1 클래스 균형화 + 3배 확장 파이프라인                    │
│                • 클래스 불균형 자동 보정 (휴식 2배 편향 제거 ➔ 25% 균등 분배)            │
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
│                • 정적 학습 가중치 (런타임 동적 가중치 배제 ➔ 신뢰도 요동 원천 차단)       │
│                • Cross-Entropy Loss with Label Smoothing (ε=0.05)                        │
│        │                                                                                 │
│        ▼                                                                                 │
│ [6. 추론 엔진] 지능형 제스처 안정화 엔진 (`GestureStabilizer`)                           │
│                • 4대 동작 클래스별(휴식, 주먹, 손펴기, 엄지) 개별 확률 실시간 동시 표기   │
│                • 인위적 확률 조작 없는 순수 신경망 예측치 기반 EMA 필터링 (α=0.25)        │
│                • 듀얼 임계치 히스테리시스 락 (진입 70%, 유지 40%)                         │
│                • 시간 디바운스 필터 (Temporal Debounce Buffer: 5 frames / 100ms)         │
│                • 무신호 데드존 스마트 에너지 세이프티 (Total AC-RMS < 35 ➔ 안전 휴식 모드)│
│                • 실시간 상태 시각화 태그 (`안정`, `전이`, `유지`, `확정`, `휴식`)         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 생체 신호 처리 및 특징 공학 (75차원 DC-Invariant Feature Engineering)

| 기법명 | 수학적 정의 / 원리 | 적용 목적 및 생체 신호적 의의 |
| :--- | :--- | :--- |
| **시계열 슬라이딩 윈도우**<br>(Sliding Window) | 윈도우 크기 $N=30$ (약 0.6초 분량) | 제스처의 동적 시간 전이 패턴을 온전히 포착하면서도 20ms 간격으로 고속 스트리밍 추론 지원 |
| **국소 중심화 변위**<br>(Centered Raw Displacement) | $x_{\text{centered}}[i] = x[i] - \mu_{\text{window}}$ | 센서 탈착/재착용 시마다 달라지는 직류 기준 전압(30~600)을 즉시 소거하고 순간 교류 변위만 5차원으로 산출 |
| **Detrended 16-FFT**<br>(비직류 고속 푸리에 변환) | $X[k] = \sum_{n=0}^{N-1} (x[n] - \mu) e^{-j\frac{2\pi}{N}kn}$<br>$\text{Mag} = \sqrt{\text{Re}^2 + \text{Im}^2}$ | 윈도우 평균을 빼는 Detrending을 통해 저주파~고주파 전 대역으로의 직류 누설(Spectral Leakage)을 차단하고 채널당 8개 주파수 대역 에너지 산출 |
| **AC-RMS (국소 표준편차)**<br>(Root Mean Square) | $\text{AC-RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2}$ | 직류 전압을 제외한 순수 근육 수축 전력(AC Power)만을 추출. 센서를 다시 착용해도 휴식/수축 구분이 완벽히 유지됨 |
| **AC-MAV (국소 평균절대편차)**<br>(Mean Absolute Value) | $\text{AC-MAV} = \frac{1}{N}\sum_{i=1}^N \|x_i - \mu\|$ | 근육 수축 신호의 순수 진폭 포락선(Envelope)을 산출하여 피로도 및 수축 강도 모델링 |
| **파형 길이 (WL)**<br>(Waveform Length) | $\text{WL} = \sum_{i=1}^{N-1} \|x_i - x_{i-1}\|$ | 차분 신호 기반으로 신호의 누적 변동량과 파형 복잡도를 표현 (수학적 100% 직류 불변) |
| **영점 교차율 (ZC)**<br>(Zero Crossing Rate) | $x_i - \mu$의 부호 변경 횟수<br>(데드존 $\|x_i - x_{i-1}\| \ge 3.0$) | 주파수 밀도를 간접 반영하며, 노이즈 불감대(Deadzone)를 두어 미세 아날로그 전자파 잡음 필터링 |
| **기울기 부호 역전율 (SSC)**<br>(Slope Sign Change) | $(x_i - x_{i-1})(x_i - x_{i+1}) \ge \epsilon$<br>($\epsilon = 4.0$, 피크/밸리 전환 카운트) | 근육 운동 단위 활동전위(MUAP)의 발화 주파수와 고주파 파동 특성을 포착. **차분곱 기반으로 100% 직류 불변** |
| **윌리슨 진폭 변동율 (WAMP)**<br>(Willison Amplitude) | $\sum_{i=1}^{N-1} \mathbb{I}(\|x_i - x_{i-1}\| \ge \theta)$<br>($\theta = 8.0$ ADC count 임계치) | 활동 전위의 발화율 및 운동 단위 동원 수준(Motor Unit Recruitment)을 포착. 수축 강도 구분력 대폭 강화 |
| **완전 DC-Invariant 공간**<br>(75차원 특징 공간) | 원시 중심화(5) + FFT(40) + 시간도메인(30) | 센서 위치 이동, 전극 재착용, 피부 임피던스 변화에도 특징 공간의 베이스라인 왜곡이 0에 수렴 |

---

### 2.2 데이터 증강, 클래스 균형화 및 정규화 (Augmentation, Balancing & Normalization)

1. **클래스 불균형 자동 균형화 (Automatic 1:1:1:1 Class Balancing) 🚀 신규**:
   - 데이터 수집 시 휴식이 처음과 끝에 2회 녹화(14초, 482개 샘플)되어 전체 데이터의 40%를 차지하던 **휴식 클래스 과적합(Class Imbalance)** 문제를 `load_dataset`에서 자동 감지.
   - 2배 많이 녹화된 휴식 샘플을 균등 샘플링(482행 ➔ 241행)하여, 4개 클래스 모두 정확히 **각 241~242행(25%씩) 완벽히 동등한 비율**로 훈련되도록 자동 보정. 휴식으로의 쏠림 현상을 원천 차단.
2. **무작위 베이스라인 시프트 (Random Baseline Shift)**:
   - 채널별로 $\pm 50$ ADC 단위의 임의 오프셋 $\Delta \in [-50, 50]$을 신호에 주입하여 학습. 전극 재착용으로 인한 기준선 표류(Baseline Drift)에 내성 부여.
3. **가우시안 노이즈 지터링 (Gaussian Noise Jittering)**:
   - $\mathcal{N}(0, \sigma^2)$ ($\sigma=2.5$) 잡음을 신호에 합성하여 피부 접촉 불량이나 전원 노이즈 내성 훈련.
4. **진폭 스케일링 (Amplitude Scaling)**:
   - 신호에 $0.85 \sim 1.20\times$ 임의 스케일을 곱하여 근육 피로도 및 수축 강도 약화 상황에서도 동일 제스처로 정확히 인식하도록 일반화.
5. **특징별 Z-Score 표준화 (Feature-wise Z-Score Standardization)**:
   - $x_{\text{norm}} = \frac{x - \mu}{\sigma + 10^{-4}}$ 공식을 통해 75개 모든 특징 열을 평균 0, 표준편차 1로 스케일링.
6. **학습-추론 분포 동기화 (`NormalizationStats`)**:
   - 학습 완료 시 산출된 평균/표준편차 벡터를 JSON(`norm_stats_emg_tcn_model.json`)으로 보존하고 추론 시 완벽 일치 적용.

---

### 2.3 딥러닝 신경망 아키텍처 (정적 1D-TCN Architecture)

#### 🥇 1D-TCN (Temporal Convolutional Network)
런타임에 입력 신호에 따라 센서 가중치를 동적으로 증감시키는 불안정한 어텐션 구조(SE-Block 등)를 배제하고, 학습 단계에서 최적화된 **정적 합성곱 커널(Stationary Learned Kernels)과 배치 정규화(BatchNorm)**만을 통과시켜 **프레임 간 신뢰도 요동(Confidence Fluctuation)을 원천 방지**한 고안정 시계열 신경망입니다.

```
[입력 텐서: Batch × 75 Features × 30 Seq]
               │
               ▼
   ┌────────────────────────────────────────────────────────┐
   │ TcnBlock 1 (d=1, c=32)                                 │
   │  ├─ Conv1d(k=3, pad=1, d=1) ➔ BatchNorm ➔ ReLU ➔ Drop │
   │  ├─ Conv1d(k=3, pad=1, d=1) ➔ BatchNorm ➔ Drop        │
   │  └─ (+) Residual Shortcut Connection                    │
   └────────────────────────────────────────────────────────┘
               │
               ▼
   ┌────────────────────────────────────────────────────────┐
   │ TcnBlock 2 (d=2, c=64)                                 │
   │  ├─ Conv1d(k=3, pad=2, d=2) ➔ BatchNorm ➔ ReLU ➔ Drop │
   │  ├─ Conv1d(k=3, pad=2, d=2) ➔ BatchNorm ➔ Drop        │
   │  └─ (+) 1x1 Conv Shortcut Connection                   │
   └────────────────────────────────────────────────────────┘
               │
               ▼
   ┌────────────────────────────────────────────────────────┐
   │ TcnBlock 3 (d=4, c=64)                                 │
   │  ├─ Conv1d(k=3, pad=4, d=4) ➔ BatchNorm ➔ ReLU ➔ Drop │
   │  ├─ Conv1d(k=3, pad=4, d=4) ➔ BatchNorm ➔ Drop        │
   │  └─ (+) Residual Shortcut Connection                    │
   └────────────────────────────────────────────────────────┘
               │
               ▼
   [AdaptiveAvgPool1d(1)]   ──> 시계열 전체를 [Batch × 64]로 글로벌 평균 풀링
               │
               ▼
   [Linear(64 ➔ 32) + ReLU + Dropout(0.15)]
               │
               ▼
   [Linear(32 ➔ 4)]         ──> 4개 클래스 최종 로짓(Logits)
```

- **정적 가중치 연산의 안정성 (런타임 동적 가중치 배제)**:
  - 런타임에 센서별 가중치를 동적으로 바꾸는 모듈은 근전도 신호의 20ms 단위 미세 생체 떨림에 반응하여 채널 가중치가 프레임마다 출렁거리고, 결과적으로 모델 확신도(Confidence)가 널뛰는 원인이 됩니다.
  - 본 파이프라인은 런타임 동적 가중을 완전히 배제하고, 학습 완료된 고정 가중치와 지수이동평균(EMA)만을 사용하여 **매우 차분하고 일관된 신뢰도**를 산출합니다.

---

### 2.4 학습 최적화 및 정규화 동역학 (Label Smoothing & Optimization)

- **라벨 스무딩 (Label Smoothing, $\epsilon=0.05$)**:
  - 원-핫 인코딩 벡터 대신 부드러운 타깃 확률 벡터($y_{\text{true}} = 0.9625, y_{\text{other}} = 0.0125$)를 적용.
  - 이론적 최솟값 $H(y^{\text{smooth}}) \approx \mathbf{0.2011}$을 기준으로 모델이 과잉 확신을 갖지 않도록 정규화하여 가중치 폭증을 방지.
- **Adam 옵티마이저 & 학습률 $0.001$**:
  - 적응형 모멘텀을 통해 손실을 $1.30$에서 $0.24$ 이하(이론상 최솟값 부근)로 안정 수렴.
- **WGPU 셰이더 가속**:
  - Rust Burn 엔진 기반 Vulkan/NVIDIA 병렬 컴퓨트 파이프라인을 통한 GPU 가속.

---

### 2.5 지능형 실시간 추론 안정화 & 4대 동작 개별 확률 모니터링 (`GestureStabilizer`)

실시간 추론 시 센서의 미세 떨림이나 손동작 전환 순간의 **결과 채터링(바운싱 현상)**을 박멸하기 위해 통계적 제어 이론과 슈미트 트리거(Schmitt Trigger)를 결합한 지능형 후처리 엔진입니다.

```
                  [모델 출력 Logits]
                          │
                          ▼
            [수치 안정 Softmax 확률 산출]
                          │
                          ▼
        ┌───────────────────────────────────┐
        │ 1. 확률 EMA 필터 (α = 0.25)        │
        │    P_ema = α*P + (1-α)*P_prev     │
        │    (인위적 확률 조작 없는 순수 추론)│
        └───────────────────────────────────┘
                          │
                          ▼
        ┌───────────────────────────────────┐
        │ 2. 듀얼 히스테리시스 락 (Hysteresis)│
        │    • 신규 진입 문턱: P >= 70%     │
        │    • 기존 유지 문턱: P_hold < 40% │
        └───────────────────────────────────┘
                          │
               ┌──────────┴──────────┐
          조건 미충족             조건 충족
               ▼                     ▼
        [기존 상태 LOCK]     ┌───────────────────────────────────┐
            (태그: 유지)      │ 3. 시간 디바운스 필터 (Debounce)   │
                             │    5프레임(100ms) 연속 유지 검증   │
                             └───────────────────────────────────┘
                                       │
                            ┌──────────┴──────────┐
                        카운트 < 5             카운트 >= 5
                            ▼                     ▼
                     [전이 후보 누적]         [최종 상태 전이]
                       (태그: 전이)            (태그: 확정)
```

1. **4대 제스처 개별 실시간 확률 표기 UI 🚀 신규**:
   - `휴식(Relax)`, `주먹 쥐기(Fist)`, `손가락 펴기(Open)`, `엄지(Thumb)`의 실시간 확률(%)을 한 줄에 동시 표시하여 근육 힘에 따른 각 클래스의 확률 변화를 직관적으로 디버깅 가능:
     ```text
     🤖 [1D-TCN ] [안정] 주먹 쥐기 (Fist)  | [휴식: 0.5% | 주먹: 94.2% | 펴기: 3.8% | 엄지: 1.5%] | RMS: 145.2
     ```
2. **인위적 확률 오버라이드 제거 & 안전 데드존 게이트 (RMS < 35.0)**:
   - 기존에 RMS < 75일 때 휴식 확률을 강제로 98%로 덮어쓰던 하드코딩을 삭제하여, **화면에 표시되는 확률은 100% 신경망의 순수 예측 결과**입니다.
   - 에너지 게이트 임계치를 35.0(완전히 힘을 뺀 무신호 데드존)으로 설정하여, 가벼운 제스처(RMS 50~70)가 휴식으로 씹히지 않도록 개선.
3. **확률 지수 이동 평균 (Probability EMA, $\alpha=0.25$)**:
   - 20ms 단위의 일시적 확률 스파이크를 완화하고 시계열적 연속성을 부여.
4. **듀얼 임계치 히스테리시스 락 (Dual-Threshold Hysteresis Locking)**:
   - **진입 임계치 ($70\%$)**: 새로운 제스처로 넘어가려면 확률이 $70\%$ 이상이어야 함.
   - **유지 임계치 ($40\%$)**: 현재 제스처를 계속 유지하는 문턱은 $40\%$로 낮추어 경계선 채터링 차단.
5. **시간 디바운스 필터 (Temporal Debounce Buffer, 5프레임 / 약 100ms)**:
   - 진입 조건을 만족하더라도 최소 5프레임(0.1초) 이상 지속될 때만 실제 전환을 승인하여 순간적인 틱(Tic) 무시.
6. **실시간 동작 상태 태그 UI**:
   - `[안정]`: 현재 제스처가 매우 높은 신뢰도로 안정 유지 중.
   - `[전이]`: 새 제스처로 전환을 시도 중인 과도기 상태 (디바운스 누적 중).
   - `[확정]`: 디바운스를 통과하여 새로운 제스처로 확정 전환된 순간.
   - `[유지]`: 애매한 신호 구간에서 이전 제스처를 락(Lock)하여 고정 중.
   - `[휴식]`: 완전 데드존(RMS < 35) 진입 시 안전 휴식 판정.

---

## 3. 특징 추출 파이프라인 (75차원 완전 직류 불변 공간)

센서 원시 신호에 의존하지 않고, 근육 수축 상태와 주파수 특성을 직류(DC) 편향 없이 추출하는 **75차원 완전 직류 불변(DC-Invariant) 특징 벡터**를 실시간 산출합니다.

$$\text{Total Features (75)} = \text{Centered Raw (5)} + \text{Detrended FFT (40)} + \text{Extended Time-Domain (30)}$$

```
[5채널 아날로그 신호] ──┬──> [1] 국소 중심화 신호 (raw - mean) (5차원)
                       ├──> [2] 16-point Detrended FFT Magnitude (채널당 8 bin × 5 = 40차원)
                       └──> [3] 생체 전자기학 확장 시간 도메인 6대 지표 (채널당 6개 × 5 = 30차원)
                                 ├─ AC-RMS (국소 표준편차): 순수 교류 근육 수축 총 파워
                                 ├─ AC-MAV (국소 평균절대편차): 순수 진폭 포락선
                                 ├─ WL (Waveform Length): 신호 누적 변동량 (차분 지표)
                                 ├─ ZC (Zero Crossing): 평균선 교차 횟수 (주파수 대용)
                                 ├─ SSC (Slope Sign Change): 기울기 부호 역전율 (MUAP 발화)
                                 └─ WAMP (Willison Amplitude): 윌리슨 진폭 변동율 (동원 수준)
```

---

## 4. 지원 딥러닝 모델 아키텍처

| 모델명 | 파일 위치 | 특징 및 구조 |
| :--- | :--- | :--- |
| **1D-TCN** (🥇 추천) | [`src/models/tcn.rs`](src/models/tcn.rs) | **Dilated Conv1d(Dilation: 1, 2, 4) + Residual + BatchNorm1d + Dropout(0.15)**<br>런타임 동적 가중치 요동 없이 일관된 정적 수용 영역 특징 추출로 안정성 극대화 |
| **1D-CNN** | [`src/models/cnn.rs`](src/models/cnn.rs) | Conv1d + BatchNorm1d + AdaptiveAvgPool1d 기반 경량 합성곱 모델 |
| **CNN + LSTM** | [`src/models/cnn_lstm.rs`](src/models/cnn_lstm.rs) | 국소적 특징 추출(Conv+BN) 후 순차적 시계열 패턴 학습(LSTM) 결합 |
| **LSTM** | [`src/models/lstm.rs`](src/models/lstm.rs) | 순수 순환 신경망 구조 |

---

## 5. 채터링 제로 & 노이즈 극복 3중 방어 체계

### 1) 1단계: 특징 레벨 (75D DC-Invariant & 3x Data Augmentation)
- 센서가 피부에서 미세하게 들뜨거나 착용 위치가 바뀌어도 중심화($x - \mu$) 및 차분($x_i - x_{i-1}$) 지표로만 구성되어 있어 베이스라인 전압 변화를 100% 흡수.
- 학습 데이터셋의 휴식 2배 편향을 1:1:1:1로 자동 균형화한 뒤, 베이스라인 시프트($\pm 50$), 가우시안 노이즈, 진폭 스케일링을 주입하여 학습.

### 2) 2단계: 모델 레벨 (Stationary TCN Weights & Label Smoothing)
- 런타임에 채널 가중치를 건드리지 않는 정적 커널로 예측 신뢰도 흔들림 차단.
- 라벨 스무딩($\epsilon=0.05$)으로 모델의 과잉 확신을 억제하여 동작 전이 구간의 돌발 오분류 방지.

### 3) 3단계: 추론 후처리 레벨 (`GestureStabilizer`)
- 확률 EMA($\alpha=0.25$) + 히스테리시스 락($70\% / 40\%$) + 100ms 디바운싱.
- 4개 동작의 실시간 확률을 있는 그대로 투명하게 노출하며, 인위적 강제 조작 없이 동작 전환 순간에만 칼같이 확정 전환.

---

## 6. 시스템 구성 및 네트워크 통신

```
┌────────────────────────────────────────┐         ┌───────────────────────────────────────┐
│        emg_client (로컬 장치)          │         │        emg_server (GPU 학습 서버)     │
│                                        │         │                                       │
│  [아두이노/센서] -> 시리얼 수신 (50Hz) │         │  • POST /api/upload (데이터 업로드)   │
│         │                              │  HTTP   │  • POST /api/train  (원격 WGPU 학습)  │
│  [특징 추출 (75D)] -> 슬라이딩 윈도우  ├────────>│  • GET  /api/status (실시간 에포크)   │
│         │                              │<────────┤  • GET  /api/download/:name           │
│  [GestureStabilizer 추론 엔진]         │         │    (.mpk 가중치 + 정규화 JSON 동시수신)│
└────────────────────────────────────────┘         └───────────────────────────────────────┘
```

---

## 7. 실행 및 사용 가이드

### 빌드 및 사전 준비
```bash
# 디버그 검사 및 단위 테스트 검증
cargo check --bins --tests
cargo test -- --nocapture

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
  ```text
  🤖 [1D-TCN ] [안정] 주먹 쥐기 (Fist)  | [휴식: 0.5% | 주먹: 94.2% | 펴기: 3.8% | 엄지: 1.5%] | RMS: 145.2
  ```

---

## 8. 파일 및 디렉토리 구조

```
dl/
├── Cargo.toml                      # 프로젝트 의존성 및 바이너리 정의
├── norm_stats_emg_tcn_model.json   # 1D-TCN 학습 데이터셋 75차원 정규화 통계 (Z-score)
├── emg_tcn_model.mpk               # 훈련 완료된 1D-TCN 모델 바이너리
├── README.md                       # 통합 기술 문서 (본 문서)
├── src/
│   ├── lib.rs                      # 코어 라이브러리 및 특징 상수 정의 (75D)
│   ├── main.rs                     # emg_client 및 GestureStabilizer 실시간 추론 엔진
│   ├── fft.rs                      # 실시간 16-FFT 및 생체 지표 6종 추출기 (75차원)
│   ├── train.rs                    # 1:1:1:1 클래스 균형화, 데이터 증강, Z-Score 표준화, Label Smoothing
│   ├── bin/
│   │   └── server.rs               # Axum 기반 원격 WGPU 학습 서버
│   └── models/
│       ├── mod.rs                  # 모델 아키텍처 열거형 및 인터페이스
│       ├── tcn.rs                  # 1D-TCN (Dilated Conv + BatchNorm + Residual + Dropout)
│       ├── cnn.rs                  # 1D-CNN (BatchNorm 적용)
│       ├── cnn_lstm.rs             # CRNN (CNN + LSTM + BatchNorm)
│       └── lstm.rs                 # 기본 LSTM 모델
└── tests/
    └── training_integration_test.rs # WGPU 모델 학습 및 75D 수렴 검증 통합 테스트
```

---

## 9. 제스처 클래스 정의 (`NUM_CLASSES: 4`)

- `Class 0`: 휴식 상태 (Relax)
- `Class 1`: 주먹 쥐기 (Fist)
- `Class 2`: 손가락 펴기 (Open Hand)
- `Class 3`: 엄지 굽히기 / 손목 (Thumb Flex / Wrist)
