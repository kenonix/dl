use burn::{
    config::Config,
    lr_scheduler::LrScheduler,
    module::{Module, Param},
    nn::{
        loss::CrossEntropyLossConfig, Dropout, DropoutConfig, Gelu, Initializer, LayerNorm,
        LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{
        activation::silu,
        backend::{AutodiffBackend, Backend},
        Distribution, Int, Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::train::EmgSeqBatch;

/// 시계열 모듈형 블록 MLP의 기본 슬라이딩 윈도우 크기 (T = 8 프레임, 타켄스 위상 공간 복원용)
pub const MODULAR_MLP_WINDOW_SIZE: usize = 8;

// =========================================================================
// [단계 1] 실시간 추론용 고정 크기 링 버퍼 (FIFO Circular Queue & Replicate Padding)
// =========================================================================
/// 실시간 스트림에서 O(1) 포인터 갱신만으로 2D 시계열 윈도우 `[1, T, C]`를 생성하는 순환 버퍼입니다.
/// - **기능 1.2 (고정 크기 링 버퍼)**: 배열 시프트(memcpy) 없이 `write_ptr = (write_ptr + 1) % T`로 순환 갱신
/// - **기능 1.3 (초기 복제 패딩)**: 첫 유효 프레임 `x_0` 유입 시 버퍼 전체를 `x_0`로 채워 초기 미분값(`dx/dt = 0`) 안정화
/// - **기능 1.4 (채널별 분포 정규화)**: `(x - μ) / (σ + 1e-8)` 표준화 내장 지원
#[derive(Clone, Debug)]
pub struct RealTimeRingBuffer {
    window_size: usize,
    num_features: usize,
    buffer: Vec<f32>,
    write_ptr: usize,
    is_initialized: bool,
    means: Vec<f32>,
    stds: Vec<f32>,
}

impl RealTimeRingBuffer {
    pub fn new(
        window_size: usize,
        num_features: usize,
        means: Option<Vec<f32>>,
        stds: Option<Vec<f32>>,
    ) -> Self {
        let t = window_size.max(1);
        let c = num_features.max(1);
        Self {
            window_size: t,
            num_features: c,
            buffer: vec![0.0f32; t * c],
            write_ptr: 0,
            is_initialized: false,
            means: means.unwrap_or_else(|| vec![0.0f32; c]),
            stds: stds.unwrap_or_else(|| vec![1.0f32; c]),
        }
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_ptr = 0;
        self.is_initialized = false;
    }

    /// 원시/미정규화 1D 프레임 `(C,)`을 입력받아 내부에서 채널별 표준화 `(x - μ) / (σ + 1e-8)`를 수행한 뒤,
    /// 시간순 정렬된 `[T * C]` 평탄화 윈도우 벡터를 반환합니다.
    pub fn push_and_get_window(&mut self, raw_frame: &[f32]) -> Vec<f32> {
        let mut norm_frame = Vec::with_capacity(self.num_features);
        for i in 0..self.num_features {
            let x = raw_frame.get(i).copied().unwrap_or(0.0);
            let m = self.means.get(i).copied().unwrap_or(0.0);
            let s = self.stds.get(i).copied().unwrap_or(1.0);
            norm_frame.push((x - m) / (s + 1e-8));
        }
        self.push_normalized_and_get_window(&norm_frame)
    }

    /// 이미 정규화된 1D 프레임 `(C,)`을 입력받아 복제 패딩(첫 프레임) 또는 O(1) 순환 갱신을 수행하고,
    /// 과거 -> 최신 순으로 정렬된 `[T * C]` 윈도우 벡터를 반환합니다.
    pub fn push_normalized_and_get_window(&mut self, norm_frame: &[f32]) -> Vec<f32> {
        let c = self.num_features;
        let t = self.window_size;

        if !self.is_initialized {
            // 기능 1.3: 첫 번째 유효 프레임 x_0로 T개 슬롯 전체를 일괄 복제 초기화 (dx/dt = 0)
            for slot in 0..t {
                let start = slot * c;
                for i in 0..c {
                    self.buffer[start + i] = norm_frame.get(i).copied().unwrap_or(0.0);
                }
            }
            self.write_ptr = 0;
            self.is_initialized = true;
        } else {
            // 기능 1.2: write_ptr 위치에 덮어쓰고 모듈러 연산으로 포인터만 이동 (O(1))
            let start = self.write_ptr * c;
            for i in 0..c {
                self.buffer[start + i] = norm_frame.get(i).copied().unwrap_or(0.0);
            }
            self.write_ptr = (self.write_ptr + 1) % t;
        }

        // 시간 순서(가장 오래된 프레임 -> 가장 최신 프레임)대로 정렬된 슬라이스 조합
        let split_offset = self.write_ptr * c;
        let mut ordered = Vec::with_capacity(t * c);
        ordered.extend_from_slice(&self.buffer[split_offset..]);
        ordered.extend_from_slice(&self.buffer[..split_offset]);
        ordered
    }

    /// 원시 프레임을 입력받아 즉시 `[1, T, C]` 3차원 텐서를 반환합니다.
    pub fn push_and_get_tensor<B: Backend>(
        &mut self,
        raw_frame: &[f32],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let ordered = self.push_and_get_window(raw_frame);
        Tensor::<B, 1>::from_floats(ordered.as_slice(), device).reshape([
            1,
            self.window_size,
            self.num_features,
        ])
    }

    /// 이미 정규화된 프레임을 입력받아 즉시 `[1, T, C]` 3차원 텐서를 반환합니다.
    pub fn push_normalized_and_get_tensor<B: Backend>(
        &mut self,
        norm_frame: &[f32],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let ordered = self.push_normalized_and_get_window(norm_frame);
        Tensor::<B, 1>::from_floats(ordered.as_slice(), device).reshape([
            1,
            self.window_size,
            self.num_features,
        ])
    }

    /// 현재 버퍼에 저장된 윈도우를 과거 -> 최신 순서의 평탄화 벡터 `[T * C]`로 반환합니다.
    pub fn get_window(&self) -> Vec<f32> {
        let c = self.num_features;
        let t = self.window_size;
        let split_offset = self.write_ptr * c;
        let mut ordered = Vec::with_capacity(t * c);
        ordered.extend_from_slice(&self.buffer[split_offset..]);
        ordered.extend_from_slice(&self.buffer[..split_offset]);
        ordered
    }

    /// 현재 버퍼에 저장된 윈도우를 `[1, T, C]` 3D 텐서로 반환합니다.
    pub fn get_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3> {
        let ordered = self.get_window();
        Tensor::<B, 1>::from_floats(ordered.as_slice(), device).reshape([
            1,
            self.window_size,
            self.num_features,
        ])
    }
}

// =========================================================================
// [단계 3] 시간 축 혼합 블록 (Temporal Mixing Block: Cross-Step Dynamics)
// =========================================================================
/// O(T^2) 어텐션 없이 채널 공유 시간 병목 MLP(FIR 미분/평활 필터)로 T개 프레임 간
/// 속도(dx/dt), 가속도(d^2x/dt^2) 및 추세 맥락을 추출합니다.
#[derive(Module, Debug)]
pub struct TemporalMixingBlock<B: Backend> {
    norm: LayerNorm<B>,
    fc1: Linear<B>,
    act: Gelu,
    drop1: Dropout,
    fc2: Linear<B>,
    drop2: Dropout,
}

impl<B: Backend> TemporalMixingBlock<B> {
    pub fn new(
        window_size: usize,
        d_model: usize,
        expansion: usize,
        dropout: f64,
        num_blocks: usize,
        device: &B::Device,
    ) -> Self {
        let d_time_hidden = window_size * expansion.max(1);
        // 기능 5.1: 잔차 분산 누적 억제를 위한 Small-Init (std = 0.02 / sqrt(2 * num_blocks))
        let small_std = 0.02 / ((2 * num_blocks.max(1)) as f64).sqrt();

        Self {
            norm: LayerNormConfig::new(d_model).init(device),
            fc1: LinearConfig::new(window_size, d_time_hidden).init(device),
            act: Gelu::new(),
            drop1: DropoutConfig::new(dropout).init(),
            fc2: LinearConfig::new(d_time_hidden, window_size)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: small_std,
                })
                .init(device),
            drop2: DropoutConfig::new(dropout).init(),
        }
    }

    /// 입력/출력 형상: `[Batch, Window(T), d_model]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // 기능 3.1: Pre-Norm 및 텐서 차원 전치 [B, T, D] -> [B, D, T]
        let h = self.norm.forward(x).swap_dims(1, 2);
        // 기능 3.2: 채널 공유 시간 병목 투영
        let h = self.fc1.forward(h);
        let h = self.act.forward(h);
        let h = self.drop1.forward(h);
        let h = self.fc2.forward(h);
        let h = self.drop2.forward(h);
        // 기능 3.3: 차원 복원 [B, D, T] -> [B, T, D]
        h.swap_dims(1, 2)
    }
}

// =========================================================================
// [단계 4] 피처 축 SwiGLU 게이팅 블록 (Channel Mixing Block: Intra-Step Feature Gating)
// =========================================================================
/// 곱셈 게이팅 `SiLU(W_gate x) ⊙ (W_val x)`을 통해 단일 블록 내부에서 피처 간 2차 다항 교차항(x_i * x_j)을
/// 직접 생성하고 불필요한 노이즈 채널을 동적으로 닫아 차단합니다.
#[derive(Module, Debug)]
pub struct SwiGLUChannelBlock<B: Backend> {
    norm: LayerNorm<B>,
    w_gate: Linear<B>,
    w_val: Linear<B>,
    drop: Dropout,
    w_out: Linear<B>,
}

impl<B: Backend> SwiGLUChannelBlock<B> {
    pub fn new(d_model: usize, dropout: f64, num_blocks: usize, device: &B::Device) -> Self {
        // 기능 4.3: 파라미터 및 FLOPs 등가 차원 설계 (2/3 * 4 * d_model, 8의 배수로 하드웨어 정렬)
        let raw_hidden = (2 * 4 * d_model) / 3;
        let d_hidden = ((raw_hidden + 7) / 8) * 8;
        let small_std = 0.02 / ((2 * num_blocks.max(1)) as f64).sqrt();

        Self {
            norm: LayerNormConfig::new(d_model).init(device),
            w_gate: LinearConfig::new(d_model, d_hidden)
                .with_bias(false)
                .init(device),
            w_val: LinearConfig::new(d_model, d_hidden)
                .with_bias(false)
                .init(device),
            drop: DropoutConfig::new(dropout).init(),
            w_out: LinearConfig::new(d_hidden, d_model)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: small_std,
                })
                .init(device),
        }
    }

    /// 입력/출력 형상: `[Batch, Window(T), d_model]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.norm.forward(x);
        let gate = silu(self.w_gate.forward(h.clone()));
        let val = self.w_val.forward(h);
        let gated = self.drop.forward(gate * val);
        self.w_out.forward(gated)
    }
}

// =========================================================================
// [단계 5] 시간+채널 결합 단위 슈퍼 블록 (LayerScale + Stochastic Depth DropPath 포함)
// =========================================================================
#[derive(Module, Debug)]
pub struct SpatioTemporalMlpBlock<B: Backend> {
    temporal_block: TemporalMixingBlock<B>,
    channel_block: SwiGLUChannelBlock<B>,
    gamma_time: Param<Tensor<B, 3>>,
    gamma_chan: Param<Tensor<B, 3>>,
    drop_path_prob: f64,
}

impl<B: Backend> SpatioTemporalMlpBlock<B> {
    pub fn new(
        window_size: usize,
        d_model: usize,
        dropout: f64,
        drop_path_prob: f64,
        layer_scale_init: f32,
        num_blocks: usize,
        device: &B::Device,
    ) -> Self {
        let gamma_init = Tensor::<B, 3>::ones([1, 1, d_model], device) * layer_scale_init;

        Self {
            temporal_block: TemporalMixingBlock::new(
                window_size,
                d_model,
                2,
                dropout,
                num_blocks,
                device,
            ),
            channel_block: SwiGLUChannelBlock::new(d_model, dropout, num_blocks, device),
            gamma_time: Param::from_tensor(gamma_init.clone()),
            gamma_chan: Param::from_tensor(gamma_init),
            drop_path_prob,
        }
    }

    /// 기능 5.2: Stochastic Depth (DropPath) — 미니배치 내 개별 샘플 단위로 블록 출력을 확률적으로 마스킹
    fn apply_drop_path(&self, x: Tensor<B, 3>, is_training: bool) -> Tensor<B, 3> {
        if !is_training || self.drop_path_prob <= 0.0 {
            return x;
        }
        let keep_prob = 1.0 - self.drop_path_prob;
        let batch_size = x.dims()[0];
        let device = x.device();
        let mask = Tensor::<B, 3>::random(
            [batch_size, 1, 1],
            Distribution::Bernoulli(keep_prob),
            &device,
        );
        (x / (keep_prob as f32)) * mask
    }

    pub fn forward_with_mode(&self, x: Tensor<B, 3>, is_training: bool) -> Tensor<B, 3> {
        // 1. 시간 축 잔차 갱신: x = x + DropPath(gamma_time * TemporalBlock(x))
        let dt = self.temporal_block.forward(x.clone()) * self.gamma_time.val();
        let x = x + self.apply_drop_path(dt, is_training);

        // 2. 채널(피처) 축 잔차 갱신: x = x + DropPath(gamma_chan * SwiGLUChannelBlock(x))
        let dc = self.channel_block.forward(x.clone()) * self.gamma_chan.val();
        x + self.apply_drop_path(dc, is_training)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_mode(x, false)
    }
}

// =========================================================================
// [단계 2 ~ 단계 6] 전체 모듈형 시계열 블록 신경망 (TemporalModularMLP)
// =========================================================================
#[derive(Config, Debug)]
pub struct EmgModularMlpConfig {
    #[config(default = 75)]
    pub d_in: usize,
    #[config(default = 8)]
    pub window_size: usize,
    #[config(default = 64)]
    pub d_model: usize,
    #[config(default = 4)]
    pub num_classes: usize,
    #[config(default = 2)]
    pub num_blocks: usize,
    #[config(default = 0.20)]
    pub dropout: f64,
    #[config(default = 0.10)]
    pub max_drop_path: f64,
    #[config(default = 1e-1)]
    pub layer_scale_init: f32,
    #[config(default = true)]
    pub causal_readout: bool,
}

#[derive(Module, Debug)]
pub struct EmgModularMlpModel<B: Backend> {
    // [단계 2] Input Stem & 학습 가능한 시점 위치 인코딩
    stem_linear: Linear<B>,
    stem_norm: LayerNorm<B>,
    stem_act: Gelu,
    pos_embed: Param<Tensor<B, 3>>,

    // [단계 5] Repeatable Backbone (SpatioTemporalMlpBlock 적층)
    backbone: Vec<SpatioTemporalMlpBlock<B>>,

    // [단계 6] Output Head (Final LayerNorm + Readout Linear)
    final_norm: LayerNorm<B>,
    head: Linear<B>,

    window_size: usize,
    d_model: usize,
    causal_readout: bool,
}

impl EmgModularMlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EmgModularMlpModel<B> {
        let n_blocks = self.num_blocks.max(1);
        let t = self.window_size.max(1);

        // 기능 2.3: 학습 가능한 시점 위치 인코딩 P ∈ R^(1 x T x d_model), std=0.02 정규분포 초기화
        let pos_init = Tensor::<B, 3>::random(
            [1, t, self.d_model],
            Distribution::Normal(0.0, 0.02),
            device,
        );

        // 기능 5.2: 첫 블록 0.0에서 마지막 블록 max_drop_path까지 선형 증가하는 DropPath 스케줄
        let mut backbone = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let dp = if n_blocks > 1 {
                self.max_drop_path * (i as f64) / ((n_blocks - 1) as f64)
            } else {
                0.0
            };
            backbone.push(SpatioTemporalMlpBlock::new(
                t,
                self.d_model,
                self.dropout,
                dp,
                self.layer_scale_init,
                n_blocks,
                device,
            ));
        }

        EmgModularMlpModel {
            stem_linear: LinearConfig::new(self.d_in, self.d_model).init(device),
            stem_norm: LayerNormConfig::new(self.d_model).init(device),
            stem_act: Gelu::new(),
            pos_embed: Param::from_tensor(pos_init),
            backbone,
            final_norm: LayerNormConfig::new(self.d_model).init(device),
            head: LinearConfig::new(self.d_model, self.num_classes).init(device),
            window_size: t,
            d_model: self.d_model,
            causal_readout: self.causal_readout,
        }
    }
}

impl<B: Backend> EmgModularMlpModel<B> {
    pub fn new(device: &B::Device, input_dim: usize, num_classes: usize) -> Self {
        EmgModularMlpConfig {
            d_in: input_dim,
            window_size: MODULAR_MLP_WINDOW_SIZE,
            d_model: 64,
            num_classes,
            num_blocks: 2,
            dropout: 0.20,
            max_drop_path: 0.10,
            layer_scale_init: 1e-1,
            causal_readout: true,
        }
        .init(device)
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// 3차원 시계열 윈도우 텐서 `[Batch, T, d_in]` 순전파 (모드 지정)
    pub fn forward_with_mode(&self, input: Tensor<B, 3>, is_training: bool) -> Tensor<B, 2> {
        let dims = input.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let feat_dim = dims[2];

        // 입력 시퀀스 길이가 모델의 window_size(T)와 다를 경우 최신 T개 프레임 슬라이싱 또는 복제 패딩
        let x_in = if seq_len == self.window_size {
            input
        } else if seq_len > self.window_size {
            input.slice([
                0..batch_size,
                (seq_len - self.window_size)..seq_len,
                0..feat_dim,
            ])
        } else {
            let last_frame = input.slice([0..batch_size, (seq_len - 1)..seq_len, 0..feat_dim]);
            last_frame.repeat_dim(1, self.window_size)
        };

        // [단계 2] Input Stem:
        // ⚠️ sEMG 신호처리 핵심 원리: 이미 채널별 Z-Score 표준화된 입력에서 Stem에 LayerNorm을 직접 걸면
        // LN(c * x) = LN(x) 스케일 불변성으로 인해 가장 중요한 물리량인 '근수축 에너지 크기(||x||_2, RMS)'가 소실됩니다.
        // 따라서 각 시점의 에너지 노름(L2 Norm)을 계산하여 LayerNorm 출력에 재결합(Norm-Preserving Stem)함으로써
        // 헤시안 조건수 안정화(등방성)와 근수축 강도 보존을 동시에 달성합니다.
        let h_stem = self.stem_linear.forward(x_in.clone());
        let energy_scale = x_in
            .powf_scalar(2.0)
            .mean_dim(2)
            .sqrt()
            .clamp(0.1, 10.0); // [Batch, T, 1] 근수축 에너지 스케일 보존
        let x = self.stem_norm.forward(h_stem) * energy_scale;
        let mut x = self.stem_act.forward(x) + self.pos_embed.val();

        // [단계 5] Repeatable Backbone
        for block in &self.backbone {
            x = block.forward_with_mode(x, is_training);
        }

        // [단계 6] Output Head: Final LayerNorm + 에너지 스케일 복원 -> Causal Slicing (최신 시점 t = -1) -> Linear Readout
        let out_energy = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .sqrt()
            .clamp(0.1, 10.0);
        let x = self.final_norm.forward(x) * out_energy;

        let feat = if self.causal_readout {
            // 기능 6.2 (Causal Slicing): 최신 프레임 x[:, -1, :]만 추출하여 위상 지연 최소화
            x.slice([
                0..batch_size,
                (self.window_size - 1)..self.window_size,
                0..self.d_model,
            ])
            .reshape([batch_size, self.d_model])
        } else {
            // 기능 6.2 (Global Mean Pooling): 구간 평균 풀링
            x.mean_dim(1).reshape([batch_size, self.d_model])
        };

        // 기능 6.3: 활성화 함수가 없는 순수 선형 타겟 투영 (Raw Logits)
        self.head.forward(feat)
    }

    /// 추론 및 검증용 결정론적 3D 순전파 `[Batch, T, d_in] -> [Batch, num_classes]`
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward_with_mode(input, false)
    }

    /// 단일 1D 스냅샷 배치 `[Batch, d_in]` 입력 시 복제 패딩으로 `[Batch, T, d_in]` 변환 후 순전파
    pub fn forward_2d(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x_3d = input.unsqueeze_dim::<3>(1).repeat_dim(1, self.window_size);
        self.forward_with_mode(x_3d, false)
    }

    pub fn forward_classification(
        &self,
        inputs: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
        is_training: bool,
    ) -> ClassificationOutput<B> {
        let output = self.forward_with_mode(inputs, is_training);
        let ce_loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        // 🚀 라벨 스무딩 (Label Smoothing, ε = 0.05):
        // 신경망이 100%(1.0) 확률을 향하여 로짓과 가중치를 과도하게 팽창시키는 과적합(Overconfidence)을 원천 억제
        let loss = if is_training {
            let eps = 0.05f32;
            let log_probs = burn::tensor::activation::log_softmax(output.clone(), 1);
            let uniform_loss = log_probs.mean().neg().reshape([1]);
            ce_loss.mul_scalar(1.0 - eps) + uniform_loss.mul_scalar(eps)
        } else {
            ce_loss
        };

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<EmgSeqBatch<B>, ClassificationOutput<B>>
    for EmgModularMlpModel<B>
{
    fn step(&self, batch: EmgSeqBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets, true);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<EmgSeqBatch<B>, ClassificationOutput<B>> for EmgModularMlpModel<B> {
    fn step(&self, batch: EmgSeqBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets, false)
    }
}

// =========================================================================
// [단계 7] 선형 웜업 결합 코사인 어닐링 스케줄러 (Cosine Annealing with Linear Warmup)
// =========================================================================
/// 기능 7.3: 전체 스텝의 초기 5~10% 구간에서는 학습률을 0(초기 미소값)에서 peak_lr까지 선형 증가시켜
/// AdamW 2차 모멘텀(v_t) 분산을 안정화하고, 이후 코사인 곡선을 따라 min_lr까지 부드럽게 감쇠시킵니다.
#[derive(Clone, Debug)]
pub struct WarmupCosineLrScheduler {
    peak_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl WarmupCosineLrScheduler {
    pub fn new(peak_lr: f64, min_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            peak_lr,
            min_lr,
            warmup_steps: warmup_steps.max(1),
            total_steps: total_steps.max(2),
            current_step: 0,
        }
    }
}

impl LrScheduler for WarmupCosineLrScheduler {
    type Record<B: Backend> = usize;

    fn step(&mut self) -> f64 {
        self.current_step += 1;
        let step = self.current_step;

        if step <= self.warmup_steps {
            // 선형 웜업 (Linear Warmup)
            let progress = (step as f64) / (self.warmup_steps as f64);
            self.min_lr + (self.peak_lr - self.min_lr) * progress
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            // 코사인 어닐링 (Cosine Annealing)
            let decay_steps = (self.total_steps - self.warmup_steps).max(1) as f64;
            let current_decay_step = (step - self.warmup_steps) as f64;
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * current_decay_step / decay_steps).cos());
            self.min_lr + (self.peak_lr - self.min_lr) * cosine
        }
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        self.current_step
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.current_step = record;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::MyBackend;
    use burn::backend::wgpu::WgpuDevice;

    #[test]
    fn test_ring_buffer_replicate_padding_and_fifo_order() {
        let mut rb = RealTimeRingBuffer::new(4, 2, None, None);

        // 1. 첫 번째 프레임 입력 시 T=4 슬롯 전체가 첫 프레임으로 복제 초기화되는지 검증 (dx/dt = 0)
        let w1 = rb.push_normalized_and_get_window(&[1.0, 2.0]);
        assert_eq!(w1, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        // 2. 순차 프레임 유입 시 오래된 프레임 -> 최신 프레임 순서로 정렬되는지 검증
        let w2 = rb.push_normalized_and_get_window(&[3.0, 4.0]);
        assert_eq!(w2, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0]);

        let w3 = rb.push_normalized_and_get_window(&[5.0, 6.0]);
        assert_eq!(w3, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let w4 = rb.push_normalized_and_get_window(&[7.0, 8.0]);
        assert_eq!(w4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // 3. write_ptr 순환 래핑 검증
        let w5 = rb.push_normalized_and_get_window(&[9.0, 10.0]);
        assert_eq!(w5, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn test_warmup_cosine_lr_scheduler() {
        let mut sched = WarmupCosineLrScheduler::new(1e-3, 1e-6, 10, 100);
        let lr1 = sched.step();
        let lr5 = {
            for _ in 0..3 {
                sched.step();
            }
            sched.step()
        };
        let lr10 = {
            for _ in 0..4 {
                sched.step();
            }
            sched.step()
        };
        assert!(lr1 > 1e-6 && lr1 < lr5);
        assert!((lr10 - 1e-3).abs() < 1e-9, "Peak LR at warmup end must be 1e-3");

        let lr55 = {
            for _ in 0..44 {
                sched.step();
            }
            sched.step()
        };
        assert!(lr55 < 1e-3 && lr55 > 1e-6);

        let lr100 = {
            for _ in 0..44 {
                sched.step();
            }
            sched.step()
        };
        assert!((lr100 - 1e-6).abs() < 1e-9, "Final LR must reach min_lr 1e-6");
    }

    #[test]
    fn test_modular_mlp_forward_shapes() {
        let device = WgpuDevice::default();
        let model = EmgModularMlpModel::<MyBackend>::new(&device, 75, 3);

        // 3D 시계열 윈도우 순전파 [Batch=4, T=8, D=75] -> [4, 3]
        let x_3d = Tensor::<MyBackend, 3>::zeros([4, MODULAR_MLP_WINDOW_SIZE, 75], &device);
        let out_3d = model.forward(x_3d);
        assert_eq!(out_3d.dims(), [4, 3]);

        // 2D 단일 프레임 호환 순전파 [Batch=2, D=75] -> [2, 3]
        let x_2d = Tensor::<MyBackend, 2>::zeros([2, 75], &device);
        let out_2d = model.forward_2d(x_2d);
        assert_eq!(out_2d.dims(), [2, 3]);
    }
}

