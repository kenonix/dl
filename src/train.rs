use crate::models::{
    EmgCnnLstmModel, EmgCnnModel, EmgLstmModel, EmgTcnModel, ModelArchitecture,
};
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::{Int, Tensor};
use burn_wgpu::{Wgpu, WgpuDevice};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

pub const SEQ_LEN: usize = 30; // 🚀 시계열 길이를 30(약 0.5초)으로 최적화하여 반응성 향상 및 과거 노이즈 중첩 방지
pub const NUM_CLASSES: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub dataset_path: String,
    pub model_arch: ModelArchitecture,
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            dataset_path: String::new(),
            model_arch: ModelArchitecture::Tcn, // 기본 모델을 1D-TCN으로 설정
            epochs: 150,
            learning_rate: 1e-3, // 🚀 기본 학습률을 0.001로 안정화
            batch_size: 32,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub epoch: usize,
    pub total_epochs: usize,
    pub loss: f32,
    pub status: String,
    pub model_type: String,
    pub model_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationStats {
    pub feature_dim: usize,
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
}

impl NormalizationStats {
    pub fn save(&self, model_name: &str) -> io::Result<()> {
        let path1 = format!("norm_stats_{}.json", model_name);
        let f1 = File::create(&path1)?;
        serde_json::to_writer_pretty(f1, self)?;

        let f2 = File::create("norm_stats.json")?;
        serde_json::to_writer_pretty(f2, self)?;
        Ok(())
    }

    pub fn load_for_model(model_name: &str) -> Option<Self> {
        let specific_path = format!("norm_stats_{}.json", model_name);
        if let Ok(f) = File::open(&specific_path) {
            if let Ok(stats) = serde_json::from_reader(f) {
                return Some(stats);
            }
        }
        if let Ok(f) = File::open("norm_stats.json") {
            if let Ok(stats) = serde_json::from_reader(f) {
                return Some(stats);
            }
        }
        None
    }

    pub fn normalize(&self, features: &[f32]) -> Vec<f32> {
        let mut norm = Vec::with_capacity(features.len());
        for (i, &val) in features.iter().enumerate() {
            let m = self.means.get(i).copied().unwrap_or(512.0);
            let s = self.stds.get(i).copied().unwrap_or(300.0);
            norm.push((val - m) / s);
        }
        norm
    }
}

use crate::fft::{EmgFftProcessor, NUM_CHANNELS, TOTAL_FEATURES_ALL};

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 123456789 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFF) as f32 / 16777216.0
    }
    fn range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

/// 데이터셋 CSV 파일에서 raw 센서값을 읽어 EmgFftProcessor를 통해 DC-Invariant 65차원 특징을 산출하고,
/// 베이스라인 시프트(±50) 및 노이즈/진폭 데이터 증강(Data Augmentation)을 적용한 후 Z-Score 표준화를 수행합니다.
pub fn load_dataset(
    file_path: &str,
) -> io::Result<(Vec<Vec<f32>>, Vec<usize>, usize, Vec<f32>, Vec<f32>)> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // 헤더 파싱
    let header = lines.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "빈 데이터셋 파일입니다.")
    })??;

    let header_cols: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
    if header_cols.len() < 7 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "잘못된 CSV 형식: 최소 7개 이상의 컬럼이 필요합니다. (발견: {})",
                header_cols.len()
            ),
        ));
    }

    let mut raw_records: Vec<([f32; NUM_CHANNELS], usize)> = Vec::new();

    for line_res in lines {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let parts: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();
        if parts.len() < 7 {
            continue;
        }

        let ch0 = parts[0].parse::<f32>().unwrap_or(0.0);
        let ch1 = parts[1].parse::<f32>().unwrap_or(0.0);
        let ch2 = parts[2].parse::<f32>().unwrap_or(0.0);
        let ch3 = parts[3].parse::<f32>().unwrap_or(0.0);
        let ch4 = parts[4].parse::<f32>().unwrap_or(0.0);
        let raw_sample = [ch0, ch1, ch2, ch3, ch4];

        let label = parts[parts.len() - 2].parse::<usize>().unwrap_or(0);
        raw_records.push((raw_sample, label));
    }

    let total_raw_rows = raw_records.len();
    if total_raw_rows == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "유효한 데이터 행이 없습니다.",
        ));
    }

    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<usize> = Vec::new();
    let mut rng = SimpleRng::new(20260921);

    // 1. 원본 신호 특징 추출 (DC-Invariant 65차원)
    let mut proc_orig = EmgFftProcessor::new();
    for &(sample, label) in &raw_records {
        let feats = proc_orig.process_sample_all(&sample);
        all_features.push(feats.to_vec());
        all_labels.push(label);
    }

    // 2. 데이터 증강 1: 착용 위치 변화 모사 (채널별 랜덤 베이스라인 시프트 ±50 + 미세 노이즈)
    let mut proc_aug1 = EmgFftProcessor::new();
    let ch_shifts = [
        rng.range(-50.0, 50.0),
        rng.range(-50.0, 50.0),
        rng.range(-50.0, 50.0),
        rng.range(-50.0, 50.0),
        rng.range(-50.0, 50.0),
    ];
    for &(sample, label) in &raw_records {
        let mut shifted = [0.0f32; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            let noise = rng.range(-2.5, 2.5);
            shifted[ch] = sample[ch] + ch_shifts[ch] + noise;
        }
        let feats = proc_aug1.process_sample_all(&shifted);
        all_features.push(feats.to_vec());
        all_labels.push(label);
    }

    // 3. 데이터 증강 2: 악력/근육 피로도 모사 (진폭 스케일링 0.85 ~ 1.20)
    let mut proc_aug2 = EmgFftProcessor::new();
    let amp_scale = rng.range(0.85, 1.20);
    for &(sample, label) in &raw_records {
        let mut scaled = [0.0f32; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            scaled[ch] = sample[ch] * amp_scale;
        }
        let feats = proc_aug2.process_sample_all(&scaled);
        all_features.push(feats.to_vec());
        all_labels.push(label);
    }

    let total_rows = all_features.len();
    let num_features = TOTAL_FEATURES_ALL;

    // 🚀 [Z-Score 표준화] 각 특징별 평균(mean) 및 표준편차(std) 계산
    let mut means = vec![0.0f32; num_features];
    let mut stds = vec![0.0f32; num_features];

    for row in &all_features {
        for k in 0..num_features {
            means[k] += row[k];
        }
    }
    for k in 0..num_features {
        means[k] /= total_rows as f32;
    }

    for row in &all_features {
        for k in 0..num_features {
            let diff = row[k] - means[k];
            stds[k] += diff * diff;
        }
    }
    for k in 0..num_features {
        stds[k] = (stds[k] / total_rows as f32).sqrt().max(1e-4);
    }

    // (x - mean) / std 정규화 적용
    let mut standardized_features = all_features;
    for row in &mut standardized_features {
        for k in 0..num_features {
            row[k] = (row[k] - means[k]) / stds[k];
        }
    }

    println!(
        "✔ [데이터 로드 & 증강 완료] 원본 {}행 ➔ 증강 후 {}행 (3배 증강, 65차원 DC-Invariant 특징)",
        total_raw_rows, total_rows
    );

    Ok((
        standardized_features,
        all_labels,
        num_features,
        means,
        stds,
    ))
}

/// 시계열 윈도우(SEQ_LEN: 30)를 생성하여 텐서 입력 데이터로 변환합니다.
pub fn create_sliding_windows(
    features: &[Vec<f32>],
    labels: &[usize],
    feature_dim: usize,
) -> (Vec<f32>, Vec<i32>, usize) {
    let total_records = features.len();
    if total_records <= SEQ_LEN {
        return (Vec::new(), Vec::new(), 0);
    }

    let mut flat_inputs = Vec::new();
    let mut targets = Vec::new();
    let mut total_samples = 0;

    for i in SEQ_LEN..total_records {
        for j in 0..SEQ_LEN {
            let rec = &features[i - SEQ_LEN + j];
            for k in 0..feature_dim {
                flat_inputs.push(rec[k]);
            }
        }
        targets.push(labels[i] as i32);
        total_samples += 1;
    }

    (flat_inputs, targets, total_samples)
}

/// 지정된 모델 아키텍처에 따라 학습을 수행하고, 저장된 모델 파일 경로를 반환합니다.
pub fn train_model<F>(
    device: &WgpuDevice,
    config: &TrainingConfig,
    mut progress_callback: F,
) -> io::Result<String>
where
    F: FnMut(TrainingProgress),
{
    let (features, labels, feature_dim, means, stds) = load_dataset(&config.dataset_path)?;
    let (flat_inputs, targets, total_samples) =
        create_sliding_windows(&features, &labels, feature_dim);

    if total_samples == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "샘플 수가 너무 적습니다 (총 {} 레코드, SEQ_LEN: {} 필요)",
                features.len(),
                SEQ_LEN
            ),
        ));
    }

    println!(
        "ℹ️ [학습 데이터셋 구성 완료] 총 윈도우 샘플: {}개 (SEQ_LEN: {}), 특징 차원: {}개 (Z-Score 표준화 적용)",
        total_samples, SEQ_LEN, feature_dim
    );

    let input_tensor: Tensor<MyAutodiffBackend, 3> =
        Tensor::<MyAutodiffBackend, 1>::from_floats(flat_inputs.as_slice(), device)
            .reshape([total_samples, SEQ_LEN, feature_dim]);

    let target_tensor: Tensor<MyAutodiffBackend, 1, Int> =
        Tensor::from_data(targets.as_slice(), device);

    let batch_size = config.batch_size;
    let epochs = config.epochs;
    let learning_rate = config.learning_rate;
    let num_batches = (total_samples + batch_size - 1) / batch_size;

    let model_file_name = config.model_arch.file_name();

    // 모델 아키텍처별 분기 학습
    match config.model_arch {
        ModelArchitecture::Tcn => {
            let mut model: EmgTcnModel<MyAutodiffBackend> =
                EmgTcnModel::new(device, feature_dim, NUM_CLASSES);
            let mut optim = AdamConfig::new().init();

            for epoch in 1..=epochs {
                let mut total_loss = 0.0;
                for b in 0..num_batches {
                    let start = b * batch_size;
                    let end = std::cmp::min(start + batch_size, total_samples);

                    let batch_input =
                        input_tensor.clone().slice([start..end, 0..SEQ_LEN, 0..feature_dim]);
                    let batch_target = target_tensor.clone().slice([start..end]);

                    let output = model.forward(batch_input);
                    let loss = burn::nn::loss::CrossEntropyLossConfig::new()
                        .init(&output.device())
                        .forward(output, batch_target);

                    let loss_item = loss.clone().into_data().to_vec::<f32>().unwrap();
                    total_loss += loss_item[0];

                    let grads = loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, &model);
                    model = optim.step(learning_rate, model, grads_params);
                }

                let avg_loss = total_loss / num_batches as f32;
                progress_callback(TrainingProgress {
                    epoch,
                    total_epochs: epochs,
                    loss: avg_loss,
                    status: "training".to_string(),
                    model_type: config.model_arch.display_name().to_string(),
                    model_file: None,
                });
            }

            let recorder = CompactRecorder::new();
            let trained = model.valid();
            recorder
                .record(trained.into_record(), model_file_name.into())
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("저장 실패: {:?}", e)))?;
        }

        ModelArchitecture::Cnn => {
            let mut model: EmgCnnModel<MyAutodiffBackend> =
                EmgCnnModel::new(device, feature_dim, NUM_CLASSES);
            let mut optim = AdamConfig::new().init();

            for epoch in 1..=epochs {
                let mut total_loss = 0.0;
                for b in 0..num_batches {
                    let start = b * batch_size;
                    let end = std::cmp::min(start + batch_size, total_samples);

                    let batch_input =
                        input_tensor.clone().slice([start..end, 0..SEQ_LEN, 0..feature_dim]);
                    let batch_target = target_tensor.clone().slice([start..end]);

                    let output = model.forward(batch_input);
                    let loss = burn::nn::loss::CrossEntropyLossConfig::new()
                        .init(&output.device())
                        .forward(output, batch_target);

                    let loss_item = loss.clone().into_data().to_vec::<f32>().unwrap();
                    total_loss += loss_item[0];

                    let grads = loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, &model);
                    model = optim.step(learning_rate, model, grads_params);
                }

                let avg_loss = total_loss / num_batches as f32;
                progress_callback(TrainingProgress {
                    epoch,
                    total_epochs: epochs,
                    loss: avg_loss,
                    status: "training".to_string(),
                    model_type: config.model_arch.display_name().to_string(),
                    model_file: None,
                });
            }

            let recorder = CompactRecorder::new();
            let trained = model.valid();
            recorder
                .record(trained.into_record(), model_file_name.into())
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("저장 실패: {:?}", e)))?;
        }

        ModelArchitecture::Lstm => {
            let mut model: EmgLstmModel<MyAutodiffBackend> =
                EmgLstmModel::new(device, feature_dim, NUM_CLASSES);
            let mut optim = AdamConfig::new().init();

            for epoch in 1..=epochs {
                let mut total_loss = 0.0;
                for b in 0..num_batches {
                    let start = b * batch_size;
                    let end = std::cmp::min(start + batch_size, total_samples);

                    let batch_input =
                        input_tensor.clone().slice([start..end, 0..SEQ_LEN, 0..feature_dim]);
                    let batch_target = target_tensor.clone().slice([start..end]);

                    let output = model.forward(batch_input);
                    let loss = burn::nn::loss::CrossEntropyLossConfig::new()
                        .init(&output.device())
                        .forward(output, batch_target);

                    let loss_item = loss.clone().into_data().to_vec::<f32>().unwrap();
                    total_loss += loss_item[0];

                    let grads = loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, &model);
                    model = optim.step(learning_rate, model, grads_params);
                }

                let avg_loss = total_loss / num_batches as f32;
                progress_callback(TrainingProgress {
                    epoch,
                    total_epochs: epochs,
                    loss: avg_loss,
                    status: "training".to_string(),
                    model_type: config.model_arch.display_name().to_string(),
                    model_file: None,
                });
            }

            let recorder = CompactRecorder::new();
            let trained = model.valid();
            recorder
                .record(trained.into_record(), model_file_name.into())
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("저장 실패: {:?}", e)))?;
        }

        ModelArchitecture::CnnLstm => {
            let mut model: EmgCnnLstmModel<MyAutodiffBackend> =
                EmgCnnLstmModel::new(device, feature_dim, NUM_CLASSES);
            let mut optim = AdamConfig::new().init();

            for epoch in 1..=epochs {
                let mut total_loss = 0.0;
                for b in 0..num_batches {
                    let start = b * batch_size;
                    let end = std::cmp::min(start + batch_size, total_samples);

                    let batch_input =
                        input_tensor.clone().slice([start..end, 0..SEQ_LEN, 0..feature_dim]);
                    let batch_target = target_tensor.clone().slice([start..end]);

                    let output = model.forward(batch_input);
                    let loss = burn::nn::loss::CrossEntropyLossConfig::new()
                        .init(&output.device())
                        .forward(output, batch_target);

                    let loss_item = loss.clone().into_data().to_vec::<f32>().unwrap();
                    total_loss += loss_item[0];

                    let grads = loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, &model);
                    model = optim.step(learning_rate, model, grads_params);
                }

                let avg_loss = total_loss / num_batches as f32;
                progress_callback(TrainingProgress {
                    epoch,
                    total_epochs: epochs,
                    loss: avg_loss,
                    status: "training".to_string(),
                    model_type: config.model_arch.display_name().to_string(),
                    model_file: None,
                });
            }

            let recorder = CompactRecorder::new();
            let trained = model.valid();
            recorder
                .record(trained.into_record(), model_file_name.into())
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("저장 실패: {:?}", e)))?;
        }
    }

    let stats = NormalizationStats {
        feature_dim,
        means,
        stds,
    };
    if let Err(e) = stats.save(model_file_name) {
        eprintln!("⚠️ 정규화 통계 저장 실패: {:?}", e);
    } else {
        println!("✔ [정규화 통계 저장] norm_stats_{}.json 저장 완료 (추론 시 완벽 일치)", model_file_name);
    }

    let saved_file = format!("{}.mpk", model_file_name);
    progress_callback(TrainingProgress {
        epoch: epochs,
        total_epochs: epochs,
        loss: 0.0,
        status: "completed".to_string(),
        model_type: config.model_arch.display_name().to_string(),
        model_file: Some(saved_file.clone()),
    });

    Ok(saved_file)
}
