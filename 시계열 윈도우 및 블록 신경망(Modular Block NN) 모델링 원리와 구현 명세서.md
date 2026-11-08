# **시계열 윈도우 및 블록 신경망(Modular Block NN) 모델링 원리와 구현 명세서**

# **1\. 신경망 모델링의 수학적·신호처리적 기초 원리**

단순 1차원 다층 퍼셉트론(Vanilla 1D MLP)을 2차원 시계열 윈도우 기반의 모듈형 블록 신경망(Stem \- Temporal/Channel Mixing Backbone \- Output Head)으로 고도화하는 과정은 다음 세 가지 수학적·공학적 기초 원리에 근거합니다.

## **1.1 타켄스의 지연 임베딩 정리(Takens' Delay Embedding)와 위상 공간 복원**

* **단일 스냅샷의 한계:** 1차원 정적 벡터 x\_t ∈ ℝ^(d\_in) 만을 입력받는 모델은 시스템이 현재 '어디에 있는가(0차 상태)'만 관측할 뿐, '어느 방향으로 얼마나 빠르게 움직이는가(1차 미분 속도 dx/dt ≈ x\_t \- x\_{t-1}, 2차 미분 가속도 d²x/dt² ≈ x\_t \- 2x\_{t-1} \+ x\_{t-2})'를 알 수 없습니다. 즉, 동일한 위치 값이라도 상승 중인지 하강 중인지 구별할 수 없어 상태 추정의 모호성(State Aliasing)이 발생합니다.  
* **지연 임베딩을 통한 궤적 복원:** 동역학계 이론(타켄스 정리)에 따르면, 단일 시점의 관측치라도 과거 T개의 지연 샘플 \[x\_{t-T+1}, ..., x\_t\]를 묶어 ℝ^(T × d\_in)의 2차원 윈도우 행렬로 구성하면 원래 시스템의 숨겨진 위상 공간(Phase Space) 궤적과 미분 특성을 완벽히 복원할 수 있습니다.

## **1.2 심층 신경망의 역전파 동역학과 잔차 항등 경로(Identity Highway)**

* **연쇄 법칙과 야코비안 곱의 붕괴:** 일자형 MLP에서 L개의 층을 거칠 때 입력 x\_0에 대한 손실 L의 기울기는 연쇄 법칙(Chain Rule)에 의해 각 층의 야코비안(Jacobian) 행렬의 곱 ∂L/∂x\_0 \= (∂L/∂x\_L) · ∏\_{l=1..L} (∂x\_l / ∂x\_{l-1}) 로 계산됩니다. 가중치 행렬의 특이값(Singular Value)이 1보다 작으면 층이 깊어질수록 기울기가 지수적으로 0에 수렴(기울기 소실)하여 앞단 레이어가 전혀 학습되지 않습니다.  
* **덧셈 분기를 통한 기울기 보존:** 블록의 입출력을 x\_l \= x\_{l-1} \+ F(x\_{l-1}) 형태의 잔차 연결(Residual Connection)로 정의하면, 미분식이 ∂x\_l / ∂x\_{l-1} \= I \+ ∂F(x\_{l-1}) / ∂x\_{l-1} 로 변환됩니다. 여기서 단위 행렬 I(상수 1)가 덧셈 항으로 존재하므로, 비선형 블록의 미분값 ∂F/∂x\_{l-1}이 0에 가까워지더라도 상위 층의 오차 신호가 단위 행렬 I를 타고 최하단 입력층까지 감쇠 없이 직접 전달됩니다.

## **1.3 손실 지형(Loss Landscape)의 곡률 안정화와 등방성 정규화**

* **내부 공변량 변화와 비등방성(Anisotropy):** 선형 계층 y \= Wx \+ b에서 가중치 기울기는 ∂L/∂W \= (∂L/∂y) · x^T 입니다. 입력 x의 특정 채널 스케일이 크거나 층을 거치며 평균·분산이 흔들리면, 손실 함수의 헤시안(Hessian) 행렬 고유값 비율(조건수, Condition Number)이 극도로 커져 손실 지형이 좁고 가파른 협곡 형태가 됩니다. 이 상태에서는 경사하강법이 최적점으로 직진하지 못하고 벽면을 따라 지그재그로 진동합니다.  
* **정규화를 통한 초구(Hypersphere) 투영:** 각 블록 진입 전에 정규화(LayerNorm 또는 RMSNorm)를 수행하면 입력 벡터의 분포가 평균 0, 분산 1의 등방성(Isotropic) 공간으로 강제 정렬되어 헤시안 조건수가 낮아지고, 높은 학습률에서도 부드럽고 빠른 수렴이 보장됩니다.

# **2\. 구현 순서별 7단계 상세 모델링 및 엔지니어링 명세**

## **단계 1: 시계열 슬라이딩 윈도우 버퍼링 및 텐서 구조화 (Temporal Windowing & Ring Buffer)**

1차원 연속 스트림 데이터를 신경망이 시간적 맥락을 해석할 수 있는 3차원 텐서 X ∈ ℝ^(B × T × d\_in) (배치 B, 윈도우 길이 T, 피처 수 d\_in)로 변환하는 데이터 파이프라인 단계입니다.

* **기능 1.1 — 세션 단위 시계열 분할 (자기상관 데이터 누수 차단):**  
  * **기초 근거:** 시계열 데이터는 인접 시점 간 강한 자기상관(Autocorrelation, Cov(x\_t, x\_{t+1}) \>\> 0)을 가집니다. 전체 시계열에서 슬라이딩 윈도우(스트라이드 1)를 먼저 추출한 뒤 무작위(Random Split)로 학습·검증 세트를 나누면, 학습 샘플 X\_t와 검증 샘플 X\_{t+1}이 T-1개의 프레임을 완전히 공유하게 됩니다. 이 경우 모델은 일반화 규칙을 학습하는 대신 경계 프레임을 단순 암기(Memorization)하여 검증 성능이 허수로 부풀려집니다.  
  * **구현 원칙:** 반드시 원본 1차원 데이터를 수집 세션(Session)이나 시간 블록 단위로 먼저 Train / Validation / Test로 격리 분할한 뒤, 분할된 각 구간 내부에서만 독립적으로 슬라이딩 윈도우를 생성해야 합니다.  
* **기능 1.2 — 고정 크기 링 버퍼(FIFO Circular Queue) 구현:**  
  * **기초 근거:** 실시간 추론 시 매 프레임마다 배열의 첫 요소를 삭제하고 끝에 새 요소를 추가(Shift 연산)하면 매번 O(T × d\_in)의 메모리 복사(memcpy) 오버헤드와 캐시 미스가 발생합니다.  
  * **구현 원칙:** 길이 T의 고정 메모리 버퍼와 쓰기 인덱스(write\_ptr)를 두고, 새 프레임 유입 시 buffer\[write\_ptr\]를 덮어쓴 뒤 write\_ptr \= (write\_ptr \+ 1\) % T 로 순환시켜 메모리 할당 비용을 O(1)로 고정합니다.  
* **기능 1.3 — 초기 경계 효과 제거를 위한 복제 패딩(Replicate Padding):**  
  * **기초 근거:** 시스템 가동 직후 버퍼가 채워지기 전(t \< T) 구간을 0(Zero Padding)으로 채우면, 0에서 실제 센서/피처 값으로 넘어가는 경계 지점에서 계단 함수(Step Function) 형태의 인위적인 급격한 변화율(고주파 스파이크 노이즈)이 발생하여 초기 추론값이 심하게 튑니다.  
  * **구현 원칙:** 첫 번째 유효 프레임 x\_0가 입력되는 순간 버퍼의 T개 슬롯 전체를 x\_0로 일괄 복제 초기화하여, 초기 상태의 시간 미분값(dx/dt)을 0으로 안정화합니다.  
* **기능 1.4 — 채널별 분포 정규화(Channel-wise Affine Standardization):**  
  * **기초 근거:** 피처마다 물리적 단위와 변동 폭이 다르면 스케일이 큰 피처가 내적 연산(w^T x)을 지배합니다.  
  * **구현 원칙:** 학습 세트에서 산출한 채널별 평균 μ ∈ ℝ^(d\_in)와 표준편차 σ ∈ ℝ^(d\_in)(또는 왜도가 심할 경우 Yeo-Johnson / Quantile 변환기)를 고정 파라미터로 저장하고, 윈도우 내 모든 프레임에 동일하게 (x \- μ) / (σ \+ ε) 변환을 적용합니다.

## **단계 2: 입력 투영부 구현 (Input Stem: Latent Space Projection)**

원시 피처 공간(d\_in)의 데이터를 백본 블록의 표준 은닉 공간(d\_model)으로 매핑하고 초기 기저(Basis)를 형성하는 어댑터 계층입니다.

* **기능 2.1 — 점별 선형 투영(Point-wise Linear Projection):**  
  * **기초 근거:** 커버의 정리(Cover's Theorem)에 따라 저차원에서 비선형적으로 얽힌 패턴은 고차원 공간으로 투영할 때 선형 분리 가능성이 높아집니다. 또한 잔차 연결(x \+ Block(x))이 성립하려면 첫 번째 블록에 진입하기 전에 텐서의 차원이 d\_model로 통일되어야 합니다.  
  * **구현 원칙:** 입력 텐서 ℝ^(B × T × d\_in)의 마지막 축에 가중치 행렬 W\_stem ∈ ℝ^(d\_in × d\_model)을 곱해 모든 시점 t \= 1..T를 동일한 변환 규칙으로 ℝ^(B × T × d\_model) 잠재 공간에 투영합니다.  
* **기능 2.2 — 초기 정규화 및 비선형 기저 활성화:**  
  * **기초 근거:** 원시 입력의 선형 결합만으로는 극단적인 입력값(Outlier)이 바로 잔차 스트림(Residual Stream)으로 유입될 수 있습니다.  
  * **구현 원칙:** Linear(d\_in, d\_model) → LayerNorm(d\_model) → GELU() 순서로 구성하여 잠재 벡터의 초기 스케일을 단위 분산으로 정렬하고 부드러운 비선형 기저를 형성합니다.  
* **기능 2.3 — 학습 가능한 시점 위치 인코딩(Learnable Positional Encoding):**  
  * **기초 근거:** 피처 축 연산 자체는 시점 순서에 불변(Permutation Invariant)하므로, 각 프레임이 '현재 시점(t=0)'인지 '과거 시점(t=-T+1)'인지에 대한 명시적 시간 좌표가 필요합니다.  
  * **구현 원칙:** 학습 가능한 파라미터 텐서 P ∈ ℝ^(1 × T × d\_model)을 작은 정규분포(std=0.02)로 초기화하여 Stem 출력 텐서에 원소별 덧셈(x \= x \+ P)으로 결합합니다.

## **단계 3: 시간 축 혼합 블록 구현 (Temporal Mixing Block: Cross-Step Dynamics)**

무거운 O(T²) 어텐션 연산 없이 순수 행렬곱만으로 T개의 프레임 간 시간적 상관관계와 다중 스케일 미분 특성(속도, 가속도, 추세)을 추출하는 모듈입니다.

* **기능 3.1 — Pre-Norm 및 텐서 차원 전치(Transpose):**  
  * **기초 근거:** 표준 선형 계층(Linear)은 항상 텐서의 가장 마지막 차원에 대해 내적을 수행합니다. 따라서 시간 축(T) 방향으로 데이터를 섞으려면 시간 차원과 채널 차원의 메모리 축을 맞바꾸어야 합니다.  
  * **구현 원칙:** 입력 X ∈ ℝ^(B × T × d\_model)에 먼저 LayerNorm(d\_model)을 적용한 뒤, transpose(1, 2)를 호출하여 텐서 형상을 ℝ^(B × d\_model × T)로 변환합니다.  
* **기능 3.2 — 채널 공유 시간 병목 투영(Channel-Shared Temporal MLP):**  
  * **기초 근거:** d\_model개의 모든 잠재 채널에 대해 동일한 시간 변환 행렬 W\_time1 ∈ ℝ^(T × d\_T)와 W\_time2 ∈ ℝ^(d\_T × T)를 적용하면, 이 가중치 행렬의 각 행(Row)은 신호처리에서의 학습 가능한 유한 임펄스 응답(FIR) 필터(예: 저역통과 평활화 필터, 1차 차분 미분 필터, 2차 차분 곡률 필터 등)로 동작합니다. 채널 간 가중치를 공유하므로 파라미터 수가 T² 수준으로 매우 작아 과적합이 방지됩니다.  
  * **구현 원칙:** 전치된 텐서에 Linear(T, d\_T) → GELU() → Dropout() → Linear(d\_T, T) → Dropout()을 적용합니다. 이때 시간 은닉 차원 d\_T는 보통 T와 같거나 2T로 설정합니다.  
* **기능 3.3 — 차원 복원 및 시간 잔차 합산:**  
  * **기초 근거:** 기존 프레임의 고유 특징(X)을 잃지 않으면서 시간적 문맥 변화량(ΔX\_time)만 보강해야 합니다.  
  * **구현 원칙:** 연산 결과를 다시 transpose(1, 2)로 ℝ^(B × T × d\_model)로 되돌린 후, 원본 입력과 합산(Y \= X \+ ΔX\_time)합니다.

## **단계 4: 피처 축 비선형 고도화 블록 구현 (Channel Mixing Block: Intra-Step Feature Gating)**

시간 문맥이 반영된 각 타임스텝 내부에서 d\_model개의 채널 간 고차 비선형 상호작용(Feature Interaction)을 추출하는 핵심 표현력 담당 모듈입니다.

* **기능 4.1 — 역병목(Inverted Bottleneck) 구조의 기하학적 분리:**  
  * **기초 근거:** d\_model 차원의 잔차 스트림을 유지하면서 블록 내부에서만 은닉 차원을 k배(보통 k=2\~4) 확장(d\_hidden \= k × d\_model)하면, 저차원에서 겹쳐 있던 결정 경계를 고차원 초평면에서 넓게 펼쳐 비선형 활성화 함수로 정교하게 마스킹한 뒤 다시 d\_model로 선형 압축할 수 있습니다.  
  * **구현 원칙:** LayerNorm(d\_model) → Linear(d\_model, k \* d\_model) → GELU() → Dropout() → Linear(k \* d\_model, d\_model) → Dropout() 후 잔차 합산을 수행합니다.  
* **기능 4.2 — SwiGLU 곱셈 게이팅(Multiplicative Gating)을 통한 2차 교차항 모델링:**  
  * **기초 근거:** 일반 선형 계층 \+ 활성화 함수 σ(W x)는 입력 변수들의 1차 선형 결합(∑ w\_i x\_i)에만 비선형성을 부여합니다. 반면 입력을 두 갈래로 나누어 원소별 곱셈(Hadamard Product, ⊙)을 수행하는 SwiGLU 구조인 SiLU(W\_gate x) ⊙ (W\_val x)는 수식 전개 시 (∑ w\_{g,i} x\_i) × (∑ w\_{v,j} x\_j) 형태가 되어, 단일 블록 내부에서 입력 피처 간의 2차 다항 교차항(Quadratic Cross-Terms, x\_i · x\_j)을 직접 생성합니다. 또한 역전파 시 미분식이 곱의 미분법 ∂(u ⊙ v)/∂x \= (∂u/∂x) ⊙ v \+ u ⊙ (∂v/∂x)를 따르므로, 불필요한 노이즈 채널은 Gate 값을 0으로 닫아 차단하고 유의미한 상호작용 신호만 증폭하는 동적 라우팅 효과를 냅니다.  
  * **구현 원칙:** 정규화된 입력을 w\_gate와 w\_val 두 개의 선형 계층에 동시에 통과시킨 뒤 F.silu(w\_gate(x)) \* w\_val(x)를 계산하고, Dropout 후 w\_out 계층으로 d\_model 차원까지 압축합니다.  
* **기능 4.3 — 파라미터 및 FLOPs 등가 차원 설계 (2/3 × 4 × d\_model):**  
  * **기초 근거:** 표준 Inverted Bottleneck(확장비 4)은 (d × 4d) 행렬 2개를 사용하여 총 8d²개의 파라미터를 가집니다. 반면 SwiGLU는 선형 행렬을 3개(w\_gate, w\_val, w\_out) 사용하므로 은닉 차원을 그대로 4d로 두면 파라미터가 12d²로 1.5배 증가합니다.  
  * **구현 원칙:** 총 파라미터 수와 연산량을 8d²로 정확히 일치시키기 위해 SwiGLU의 은닉 차원을 d\_hidden \= int((2 / 3\) \* (4 \* d\_model)) (약 2.67 × d\_model, 하드웨어 정렬을 위해 8 또는 16의 배수로 반올림)로 설정하고, 게이팅 구조에서 불필요한 선형 편향(bias=False)을 제거합니다.

## **단계 5: 반복 백본 적층 및 잔차 경로 안정화 (Repeatable Backbone & Deep Signal Propagation)**

단계 3(Temporal Mixing)과 단계 4(Channel Mixing)를 결합한 단위 슈퍼 블록을 N개 적층하고, 깊은 층에서의 신호 분산 폭주를 제어하는 단계입니다.

* **기능 5.1 — 잔차 분산 누적 억제와 영점/소규모 가중치 초기화(Zero/Small-Init):**  
  * **기초 근거:** Pre-Norm 잔차 네트워크 x\_L \= x\_0 \+ ∑\_{l=1..L} F\_l(x\_{l-1}) 에서 각 블록 출력 F\_l의 분산이 σ²이고 서로 독립에 가깝다면, L번째 블록을 통과한 신호의 분산은 Var(x\_L) ≈ Var(x\_0) \+ L · σ² 로 층 수 L에 비례해 선형 증가합니다. 이로 인해 깊은 층에서는 학습 초기부터 신호 스케일이 비대해져 기울기가 불안정해집니다.  
  * **구현 원칙:** 각 잔차 블록의 가장 마지막 선형 계층(fc2 또는 w\_out)의 가중치를 표준편차 0.02 / √(2 × N\_blocks) 의 작은 정규분포(또는 완전한 0)로 초기화합니다. 이렇게 하면 학습 시작 시점에 모든 블록의 출력 F\_l(x) ≈ 0 이 되어 전체 깊은 네트워크가 완벽한 항등 함수(x\_L ≈ x\_0) 상태에서 출발하며, 학습이 진행됨에 따라 필요한 블록부터 점진적으로 비선형성을 획득합니다.  
* **기능 5.2 — LayerScale 및 Stochastic Depth (DropPath):**  
  * **기초 근거:** 블록이 많아질수록 모든 채널이 동일한 속도로 잔차 스트림에 더해지면 특정 지배적 채널에 과적합되기 쉽습니다.  
  * **구현 원칙:**  
    * **LayerScale:** 잔차 합산 직전 블록 출력에 학습 가능한 채널별 미소 스케일 벡터 γ\_scale ∈ ℝ^(d\_model)(초기값 1e-4)을 원소별로 곱해(x \+ γ\_scale ⊙ F(x)) 채널별 학습 속도를 자율 조절합니다.  
    * **DropPath:** 학습 시 미니배치 내 개별 샘플 단위로 확률 p\_l (첫 블록 0에서 마지막 블록 p\_max까지 선형 증가)만큼 블록 연산 F\_l(x) 전체를 통째로 0으로 마스킹하여, 서로 다른 깊이를 가진 2^N개의 부분 네트워크 앙상블 효과를 유도합니다.  
* **기능 5.3 — 피라미드 차원 전환 시 선형 투영 숏컷(Projection Shortcut):**  
  * **기초 근거:** 만약 연산량 절감이나 정보 병목 유도를 위해 스테이지마다 은닉 차원을 줄이는 피라미드 구조(예: 256 → 128)를 채택할 경우, x ∈ ℝ^(d\_in)와 F(x) ∈ ℝ^(d\_out)의 차원이 달라 덧셈이 불가능합니다.  
  * **구현 원칙:** 차원이 변하는 전환 블록의 스킵 커넥션 경로에 편향이 없는 선형 투영 행렬 W\_proj ∈ ℝ^(d\_in × d\_out)를 배치하여 Y \= W\_proj(X) \+ F(X) 로 부분공간(Subspace) 차원을 정렬합니다.

## **단계 6: 시점 집약 및 출력 예측부 구현 (Output Head: Temporal Aggregation & Readout)**

백본을 통과한 3차원 특징 텐서 ℝ^(B × T × d\_model)의 최종 스케일을 보정하고 시간 차원을 응축하여 목표 예측값 ℝ^(B × d\_out)을 산출하는 단계입니다.

* **기능 6.1 — 종단 정규화(Final LayerNorm)의 필수 배치:**  
  * **기초 근거:** Pre-Norm 아키텍처에서는 각 블록의 입력만 정규화될 뿐, 블록의 출력은 정규화 없이 잔차 스트림에 계속 합산(x\_L \= x\_0 \+ ∑ F\_l)됩니다. 따라서 마지막 블록을 빠져나온 텐서 x\_L은 누적된 분산으로 인해 스케일이 크게 부풀어 있으며, 이를 곧바로 출력 선형 계층에 통과시키면 초기 로짓(Logit) 값이 과도하게 커져 손실 함수(특히 Softmax 교차 엔트로피)의 기울기가 포화(Saturation)됩니다.  
  * **구현 원칙:** 마지막 백본 블록 직후, 시점 집약 또는 출력 선형 계층 진입 전에 반드시 독립된 LayerNorm(d\_model)을 배치하여 누적된 잔차 스트림을 다시 평균 0, 분산 1로 정규화해야 합니다.  
* **기능 6.2 — 태스크 특성에 따른 시간 축 응축(Causal Indexing vs Global Pooling):**  
  * **기초 근거:** 백본을 거친 텐서 ℝ^(B × T × d\_model)에서 T개의 시점 벡터를 단일 벡터 ℝ^(B × d\_model)로 줄이는 방식은 목표 태스크의 위상 지연(Phase Lag) 허용도에 따라 결정됩니다.  
  * **구현 원칙:**  
    * **실시간 제어 및 즉각적 상태 추정(Causal Slicing):** 가장 최신 프레임인 x\[:, \-1, :\] 슬라이스만 추출합니다. 앞선 Temporal Mixing 블록에서 이미 과거 프레임(0 \~ T-2)의 미분·문맥 정보가 최신 프레임(-1)의 잠재 벡터로 모두 흡수되었으므로, 시간 평균으로 인한 위상 지연(둔화 현상) 없이 가장 민첩한 예측이 가능합니다.  
    * **구간 전체의 클래스 분류 및 안정적 인식(Global Mean Pooling):** 윈도우 전체가 하나의 정적 상태를 나타낼 때는 시간 축 평균 x.mean(dim=1)을 취해 개별 프레임의 고주파 측정 노이즈를 상쇄합니다.  
* **기능 6.3 — 순수 선형 타겟 투영(Unactivated Readout Linear):**  
  * **기초 근거:** 출력층 끝에 Sigmoid나 Softmax를 직접 붙이면 극단값에서 부동소수점 언더플로우/오버플로우가 발생합니다.  
  * **구현 원칙:** Output Head의 마지막은 활성화 함수가 없는 순수 Linear(d\_model, d\_out)로 끝내어 원시 로짓(Raw Logits) 또는 연속 실수값을 출력하고, 확률 변환은 수치적으로 안정화된 손실 함수(CrossEntropyLoss 내부의 Log-Sum-Exp 트릭 등)에 위임합니다.

## **단계 7: 정규화 스케줄링 및 실시간 추론 루프 결합 (Optimization Geometry & Real-Time Deployment)**

설계된 아키텍처가 넓고 평탄한 최적점(Flat Minima)에 수렴하도록 파라미터 그룹별 최적화를 구성하고, 학습-추론 간 불일치 없이 배포하는 단계입니다.

* **기능 7.1 — 배치 독립적 정규화(LayerNorm / RMSNorm)를 통한 추론 결정론 보장:**  
  * **기초 근거:** BatchNorm1d는 학습 시 미니배치(B \> 1\) 샘플들 간의 평균·분산을 사용하고 추론 시에는 학습 중 누적한 이동평균(Running Mean/Var)을 사용합니다. 그러나 시계열 스트림이나 온디바이스 실시간 추론(B \= 1)에서는 미니배치 통계량과 단일 샘플의 분포가 어긋나는 치명적인 학습-추론 불일치(Train-Test Discrepancy)가 발생합니다.  
  * **구현 원칙:** 전 구간에서 배치 축(B)과 무관하게 단일 샘플의 채널 축(d\_model) 내부에서만 정규화를 수행하는 LayerNorm 또는 평균 차감을 생략해 연산 속도를 높인 RMSNorm(x / √(mean(x²) \+ ε) \* γ)을 사용하여, B=1 실시간 추론에서도 학습 시점과 수학적으로 100% 동일한 출력을 보장합니다.  
* **기능 7.2 — 분리된 가중치 감쇠(AdamW)와 파라미터 그룹 이원화(Weight Decay Exclusion):**  
  * **기초 근거:** Adam 옵티마이저에 일반 L2 정규화를 쓰면 기울기 제곱의 이동평균(v\_t)으로 나뉘어 기울기가 큰 가중치일수록 정규화 효과가 사라집니다. 따라서 가중치 감쇠 항을 분리한 AdamW를 써야 합니다. 이때 모든 파라미터에 일괄적으로 Weight Decay를 적용하면, LayerNorm의 스케일 파라미터(γ, 기본값 1)와 편향(bias), 위치 인코딩(P)까지 0으로 수축시켜 신경망의 표현력을 심각하게 훼손합니다.  
  * **구현 원칙:** 옵티마이저에 파라미터를 넘길 때 반드시 2차원 이상의 행렬 가중치(Linear.weight)만 decay\_group(예: weight\_decay=1e-2)에 넣고, 1차원 이하의 모든 텐서(bias, LayerNorm.weight, LayerNorm.bias, LayerScale, Positional Encoding)는 no\_decay\_group(weight\_decay=0.0)으로 엄격히 분리합니다.  
* **기능 7.3 — 선형 웜업 결합 코사인 어닐링(Cosine Annealing with Linear Warmup):**  
  * **기초 근거:** 학습 시작 직후(Step 1\~100)에는 AdamW의 2차 모멘텀(v\_t) 표본이 부족해 분산 추정치가 0에 가까워지며, 이로 인해 유효 스텝 크기가 비정상적으로 증폭되어 초기 가중치가 튀는 현상이 발생합니다.  
  * **구현 원칙:** 전체 학습 스텝의 첫 5\~10% 구간에서는 학습률을 0에서 목표치(예: 1e-3)까지 선형적으로 증가(Linear Warmup)시켜 모멘텀 통계량이 안정화될 시간을 확보하고, 이후 코사인 곡선을 따라 최소 학습률(예: 1e-6)까지 부드럽게 감쇠시켜 좁은 지역 최적점을 탈출하고 평탄한 최적점에 안착시킵니다.

# **3\. 단계별 텐서 형상(Shape) 및 핵심 수식 요약표**

| 단계 | 모듈 명칭 | 입력 텐서 형상 | 출력 텐서 형상 | 핵심 연산 및 수식 메커니즘 |
| :---- | :---- | :---- | :---- | :---- |
| 단계 1 | Sliding Ring Buffer | ℝ^(d\_in) (단일 프레임) | ℝ^(B × T × d\_in) | FIFO 순환 큐 \+ 초기 프레임 복제 패딩 \+ 채널별 표준화 |
| 단계 2 | Input Stem | ℝ^(B × T × d\_in) | ℝ^(B × T × d\_model) | H\_0 \= GELU(LN(X · W\_stem \+ b)) \+ P\_pos |
| 단계 3 | Temporal Mixing | ℝ^(B × T × d\_model) | ℝ^(B × T × d\_model) | H' \= H \+ Transpose(MLP\_time(Transpose(LN(H)))) |
| 단계 4 | Channel Mixing | ℝ^(B × T × d\_model) | ℝ^(B × T × d\_model) | H'' \= H' \+ γ ⊙ ((SiLU(LN(H')W\_g) ⊙ LN(H')W\_v) W\_o) |
| 단계 5 | Repeatable Backbone | ℝ^(B × T × d\_model) | ℝ^(B × T × d\_model) | 단계 3 \+ 단계 4 결합 블록을 N회 반복 적층 (Small-Init 적용) |
| 단계 6 | Output Head | ℝ^(B × T × d\_model) | ℝ^(B × d\_out) | Y \= Slice\_{t=-1}(Final\_LN(H\_N)) · W\_head \+ b\_head |
| 단계 7 | Optimization Loop | ℝ^(B × d\_out) | 스칼라 Loss / 추론값 | AdamW (1D 파라미터 WD 제외) \+ Warmup Cosine 스케줄링 |

# **4\. 전 과정 통합 레퍼런스 구현 코드**

## **4.1 PyTorch 기반 링 버퍼 \+ Temporal-SwiGLU 블록 신경망 \+ 옵티마이저 분리 통합 코드**

```py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# [단계 1] 실시간 추론용 고정 크기 링 버퍼 (FIFO & 복제 패딩)
# ==========================================
class RealTimeRingBuffer:
    def __init__(self, window_size: int, num_features: int, mean: np.ndarray = None, std: np.ndarray = None):
        self.T = window_size
        self.C = num_features
        self.buffer = np.zeros((self.T, self.C), dtype=np.float32)
        self.write_ptr = 0
        self.is_initialized = False
        self.mean = mean if mean is not None else np.zeros(self.C, dtype=np.float32)
        self.std = std if std is not None else np.ones(self.C, dtype=np.float32)

    def push_and_get_window(self, frame: np.ndarray) -> torch.Tensor:
        """단일 1D 프레임(C,)을 입력받아 정규화된 2D 윈도우 텐서(1, T, C)를 반환"""
        norm_frame = (frame.astype(np.float32) - self.mean) / (self.std + 1e-8)

        # 초기 경계 스파이크 방지를 위한 첫 프레임 복제 패딩(Replicate Padding)
        if not self.is_initialized:
            self.buffer[:] = norm_frame
            self.write_ptr = 0
            self.is_initialized = True
        else:
            self.buffer[self.write_ptr] = norm_frame
            self.write_ptr = (self.write_ptr + 1) % self.T

        # 시간 순서(가장 오래된 프레임 -> 가장 최신 프레임)대로 정렬된 뷰 생성
        ordered = np.concatenate(
            [self.buffer[self.write_ptr:], self.buffer[:self.write_ptr]], axis=0
        )
        return torch.from_numpy(ordered).unsqueeze(0)  # Shape: (1, T, C)


# ==========================================
# [단계 5 보조] Stochastic Depth (DropPath)
# ==========================================
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


# ==========================================
# [단계 3] 시간 축 혼합 블록 (Temporal Mixing Block)
# ==========================================
class TemporalMixingBlock(nn.Module):
    def __init__(self, window_size: int, d_model: int, expansion: int = 2, dropout: float = 0.1, num_blocks: int = 4):
        super().__init__()
        d_time_hidden = window_size * expansion
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(window_size, d_time_hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_time_hidden, window_size)
        self.drop2 = nn.Dropout(dropout)

        # 잔차 분산 누적 억제를 위한 Small-Init
        nn.init.normal_(self.fc2.weight, std=0.02 / math.sqrt(2 * num_blocks))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        h = self.norm(x)
        h = h.transpose(1, 2)          # (B, d_model, T)
        h = self.drop1(self.act(self.fc1(h)))
        h = self.drop2(self.fc2(h))
        h = h.transpose(1, 2)          # (B, T, d_model)
        return h


# ==========================================
# [단계 4] 피처 축 SwiGLU 게이팅 블록 (Channel Mixing Block)
# ==========================================
class SwiGLUChannelBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, num_blocks: int = 4):
        super().__init__()
        # 파라미터 등가 차원 계산 (8/3 * d_model, 8의 배수로 정렬)
        raw_hidden = int((2 / 3) * (4 * d_model))
        d_hidden = ((raw_hidden + 7) // 8) * 8

        self.norm = nn.LayerNorm(d_model)
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_val = nn.Linear(d_model, d_hidden, bias=False)
        self.drop = nn.Dropout(dropout)
        self.w_out = nn.Linear(d_hidden, d_model, bias=False)

        # 잔차 분산 누적 억제를 위한 Small-Init
        nn.init.normal_(self.w_out.weight, std=0.02 / math.sqrt(2 * num_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        h = self.norm(x)
        gated = F.silu(self.w_gate(h)) * self.w_val(h)
        return self.w_out(self.drop(gated))


# ==========================================
# [단계 5] 시간+채널 결합 단위 슈퍼 블록 (LayerScale + DropPath 포함)
# ==========================================
class SpatioTemporalMlpBlock(nn.Module):
    def __init__(
        self,
        window_size: int,
        d_model: int,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-4,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.temporal_block = TemporalMixingBlock(window_size, d_model, expansion=2, dropout=dropout, num_blocks=num_blocks)
        self.channel_block = SwiGLUChannelBlock(d_model, dropout=dropout, num_blocks=num_blocks)

        self.gamma_time = nn.Parameter(layer_scale_init * torch.ones(d_model))
        self.gamma_chan = nn.Parameter(layer_scale_init * torch.ones(d_model))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 시간 축 잔차 갱신
        x = x + self.drop_path(self.gamma_time * self.temporal_block(x))
        # 2. 채널(피처) 축 잔차 갱신
        x = x + self.drop_path(self.gamma_chan * self.channel_block(x))
        return x


# ==========================================
# [단계 2 ~ 단계 6] 전체 모듈형 시계열 블록 신경망
# ==========================================
class TemporalModularMLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        window_size: int = 8,
        d_model: int = 128,
        d_out: int = 5,
        num_blocks: int = 4,
        dropout: float = 0.1,
        max_drop_path: float = 0.1,
        causal_readout: bool = True,
    ):
        super().__init__()
        self.causal_readout = causal_readout

        # [단계 2] Input Stem & 학습 가능한 시점 위치 인코딩
        self.stem = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        # [단계 5] Repeatable Backbone (선형 증가하는 DropPath 스케줄 적용)
        dpr = [x.item() for x in torch.linspace(0, max_drop_path, num_blocks)]
        self.backbone = nn.ModuleList([
            SpatioTemporalMlpBlock(
                window_size=window_size,
                d_model=d_model,
                dropout=dropout,
                drop_path=dpr[i],
                num_blocks=num_blocks,
            )
            for i in range(num_blocks)
        ])

        # [단계 6] Output Head (Pre-Norm 누적 분산 보정을 위한 Final LayerNorm 필수)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in)
        x = self.stem(x) + self.pos_embed

        for block in self.backbone:
            x = block(x)

        x = self.final_norm(x)

        # 실시간 제어용 최신 시점 슬라이싱 vs 구간 평균 풀링
        if self.causal_readout:
            feat = x[:, -1, :]       # (B, d_model) - 위상 지연 최소화
        else:
            feat = x.mean(dim=1)     # (B, d_model) - 노이즈 평활화

        return self.head(feat)       # (B, d_out)


# ==========================================
# [단계 7] 가중치 감쇠(Weight Decay) 이원화 옵티마이저 빌더
# ==========================================
def build_adamw_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-2):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1차원 이하 파라미터(bias, LayerNorm, LayerScale) 및 위치 임베딩은 Weight Decay 제외
        if param.ndim <= 1 or "pos_embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
```

## **4.2 Rust (Burn 프레임워크) 기반 2D 윈도우 Temporal-SwiGLU 블록 구현 템플릿**

```
use burn::{
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{activation::silu, backend::Backend, Tensor},
};

/// [단계 3 & 4] 시간 축 혼합 + SwiGLU 채널 혼합이 결합된 모듈형 블록
#[derive(Module, Debug)]
pub struct SpatioTemporalMlpBlock<B: Backend> {
    // Temporal Mixing 구성 요소
    norm_time: LayerNorm<B>,
    time_fc1: Linear<B>,
    time_act: Gelu,
    time_drop1: Dropout,
    time_fc2: Linear<B>,
    time_drop2: Dropout,

    // SwiGLU Channel Mixing 구성 요소
    norm_chan: LayerNorm<B>,
    w_gate: Linear<B>,
    w_val: Linear<B>,
    chan_drop: Dropout,
    w_out: Linear<B>,
}

impl<B: Backend> SpatioTemporalMlpBlock<B> {
    pub fn new(
        window_size: usize,
        d_model: usize,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let d_time_hidden = window_size * 2;
        let raw_hidden = (2 * 4 * d_model) / 3;
        let d_chan_hidden = ((raw_hidden + 7) / 8) * 8;

        Self {
            norm_time: LayerNormConfig::new(d_model).init(device),
            time_fc1: LinearConfig::new(window_size, d_time_hidden).init(device),
            time_act: Gelu::new(),
            time_drop1: DropoutConfig::new(dropout).init(),
            time_fc2: LinearConfig::new(d_time_hidden, window_size).init(device),
            time_drop2: DropoutConfig::new(dropout).init(),

            norm_chan: LayerNormConfig::new(d_model).init(device),
            w_gate: LinearConfig::new(d_model, d_chan_hidden).with_bias(false).init(device),
            w_val: LinearConfig::new(d_model, d_chan_hidden).with_bias(false).init(device),
            chan_drop: DropoutConfig::new(dropout).init(),
            w_out: LinearConfig::new(d_chan_hidden, d_model).with_bias(false).init(device),
        }
    }

    /// 입력 형상: [Batch, Window(T), d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // 1. Temporal Mixing: [B, T, D] -> Transpose -> [B, D, T] -> Linear -> Transpose -> [B, T, D]
        let res_time = x.clone();
        let h_time = self.norm_time.forward(x).swap_dims(1, 2);
        let h_time = self.time_fc1.forward(h_time);
        let h_time = self.time_act.forward(h_time);
        let h_time = self.time_drop1.forward(h_time);
        let h_time = self.time_fc2.forward(h_time);
        let h_time = self.time_drop2.forward(h_time).swap_dims(1, 2);
        let x = res_time + h_time;

        // 2. SwiGLU Channel Mixing: [B, T, D]
        let res_chan = x.clone();
        let h_chan = self.norm_chan.forward(x);
        let gate = silu(self.w_gate.forward(h_chan.clone()));
        let val = self.w_val.forward(h_chan);
        let gated = self.chan_drop.forward(gate * val);
        let out_chan = self.w_out.forward(gated);

        res_chan + out_chan
    }
}
```

