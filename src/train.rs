use crate::models::{EmgCnnLstmModel, EmgCnnModel, EmgLstmModel, ModelArchitecture};
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

pub const SEQ_LEN: usize = 60;
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
            model_arch: ModelArchitecture::Lstm,
            epochs: 150,
            learning_rate: 1e-2,
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

/// 데이터셋 CSV 파일에서 특징과 라벨을 파싱하여 반환합니다.
/// (원시 5채널 CSV 및 FFT가 포함된 45채널 CSV 모두 자동 지원)
pub fn load_dataset(file_path: &str) -> io::Result<(Vec<Vec<f32>>, Vec<usize>, usize)> {
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
            format!("잘못된 CSV 형식: 최소 7개 이상의 컬럼이 필요합니다. (발견: {})", header_cols.len()),
        ));
    }

    // 마지막 2개는 label, action_name 이므로 그 앞의 모든 컬럼이 특징임
    let num_features = header_cols.len() - 2;

    let mut all_features = Vec::new();
    let mut all_labels = Vec::new();

    for line_res in lines {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let parts: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();
        if parts.len() != header_cols.len() {
            continue;
        }

        let mut row_feats = Vec::with_capacity(num_features);
        for i in 0..num_features {
            let val = parts[i].parse::<f32>().unwrap_or(0.0);
            // 0~1023 범위를 0~1.0 사이로 정규화
            row_feats.push(val / 1023.0);
        }

        let label = parts[num_features].parse::<usize>().unwrap_or(0);
        all_features.push(row_feats);
        all_labels.push(label);
    }

    Ok((all_features, all_labels, num_features))
}

/// 시계열 윈도우(SEQ_LEN: 60)를 생성하여 텐서 입력 데이터로 변환합니다.
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
/// progress_callback은 매 에포크마다 현재 진행 상황을 보고받습니다.
pub fn train_model<F>(
    device: &WgpuDevice,
    config: &TrainingConfig,
    mut progress_callback: F,
) -> io::Result<String>
where
    F: FnMut(TrainingProgress),
{
    let (features, labels, feature_dim) = load_dataset(&config.dataset_path)?;
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
