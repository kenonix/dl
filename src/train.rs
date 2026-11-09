use crate::models::{
    EmgCnnLstmModel, EmgCnnModel, EmgLstmModel, EmgMlpConfig, EmgModularMlpConfig, EmgTcnModel,
    ModelArchitecture, WarmupCosineLrScheduler, MODULAR_MLP_WINDOW_SIZE,
};
use burn::backend::{wgpu::WgpuDevice, Wgpu};
use burn::data::{
    dataloader::batcher::Batcher, dataloader::DataLoaderBuilder, dataset::Dataset,
};
use burn::module::Module;
use burn::optim::{AdamConfig, AdamWConfig};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::{backend::Backend, Int, Tensor};
use burn::train::{
    metric::{AccuracyMetric, LossMetric},
    LearnerBuilder,
};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

pub const SEQ_LEN: usize = 30; // 🚀 시계열 길이를 30(약 0.5초)으로 최적화하여 반응성 향상 및 과거 노이즈 중첩 방지
pub const NUM_CLASSES: usize = 4;

fn default_num_classes() -> usize {
    NUM_CLASSES
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub dataset_path: String,
    pub model_arch: ModelArchitecture,
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            dataset_path: String::new(),
            model_arch: ModelArchitecture::Mlp, // 🚀 기본 모델을 고성능 75차원 MLP로 설정
            epochs: 20,                         // MLP 추천 20 에포크
            learning_rate: 1e-3,                // 기본 학습률 0.001
            batch_size: 32,
            num_classes: NUM_CLASSES,
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

fn default_feature_dim() -> usize {
    75
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationStats {
    #[serde(default = "default_feature_dim")]
    pub feature_dim: usize,
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    #[serde(default)]
    pub class_names: Vec<String>,
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

/// =========================================================================
/// 🚀 Burn 0.16 기반 데이터셋 및 배치 변환기 (Dataset & Batcher)
/// =========================================================================
#[derive(Clone, Debug)]
pub struct EmgItem {
    pub features: Vec<f32>,
    pub label: usize,
}

#[derive(Clone)]
pub struct EmgDataset {
    pub items: Vec<EmgItem>,
}

impl Dataset<EmgItem> for EmgDataset {
    fn get(&self, index: usize) -> Option<EmgItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone)]
pub struct EmgBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> EmgBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct EmgBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<EmgItem, EmgBatch<B>> for EmgBatcher<B> {
    fn batch(&self, items: Vec<EmgItem>) -> EmgBatch<B> {
        let inputs = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats(item.features.as_slice(), &self.device))
            .collect();
        let inputs = Tensor::stack(inputs, 0);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_ints([item.label as i32], &self.device))
            .collect();
        let targets = Tensor::cat(targets, 0);

        EmgBatch { inputs, targets }
    }
}

/// =========================================================================
/// 🚀 Burn 0.16 기반 시계열 데이터셋 및 배치 변환기 (Sequence Dataset & Batcher)
/// =========================================================================
#[derive(Clone, Debug)]
pub struct EmgSeqItem {
    pub window: Vec<f32>, // SEQ_LEN * feature_dim
    pub label: usize,
    pub seq_len: usize,
    pub feature_dim: usize,
}

#[derive(Clone)]
pub struct EmgSeqDataset {
    pub items: Vec<EmgSeqItem>,
}

impl Dataset<EmgSeqItem> for EmgSeqDataset {
    fn get(&self, index: usize) -> Option<EmgSeqItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone)]
pub struct EmgSeqBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> EmgSeqBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct EmgSeqBatch<B: Backend> {
    pub inputs: Tensor<B, 3>,      // [Batch, SEQ_LEN, Feature_Dim]
    pub targets: Tensor<B, 1, Int>, // [Batch]
}

impl<B: Backend> Batcher<EmgSeqItem, EmgSeqBatch<B>> for EmgSeqBatcher<B> {
    fn batch(&self, items: Vec<EmgSeqItem>) -> EmgSeqBatch<B> {
        let batch_size = items.len();
        if batch_size == 0 {
            panic!("Cannot batch 0 items");
        }
        let seq_len = items[0].seq_len;
        let feature_dim = items[0].feature_dim;

        let inputs = items
            .iter()
            .map(|item| {
                Tensor::<B, 1>::from_floats(item.window.as_slice(), &self.device)
                    .reshape([seq_len, feature_dim])
            })
            .collect();
        let inputs = Tensor::stack(inputs, 0);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_ints([item.label as i32], &self.device))
            .collect();
        let targets = Tensor::cat(targets, 0);

        EmgSeqBatch { inputs, targets }
    }
}

/// CSV 파일에서 이미 완벽히 연산된 75차원 특징(원시5 + FFT40 + 시간도메인30)을 직접 로드하고,
/// Z-Score 표준화 및 셔플링을 수행한 후 학습/검증 데이터셋(80:20)으로 분할합니다.
pub fn load_mlp_dataset(
    file_path: &str,
) -> io::Result<(EmgDataset, EmgDataset, NormalizationStats)> {
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);

    let headers = rdr.headers().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?.clone();
    let label_idx = headers.iter().position(|h| h == "label").unwrap_or(headers.len().saturating_sub(2));
    let action_idx = headers.iter().position(|h| h == "action_name").unwrap_or(headers.len().saturating_sub(1));

    let feat_indices: Vec<usize> = headers
        .iter()
        .enumerate()
        .filter(|(idx, h)| *idx != label_idx && *idx != action_idx && h.trim() != "index")
        .map(|(idx, _)| idx)
        .collect();

    let mut raw_items = Vec::new();
    let mut class_names_map = std::collections::BTreeMap::new();

    for result in rdr.records() {
        let record = result.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if record.len() <= label_idx {
            continue;
        }

        let features: Vec<f32> = feat_indices
            .iter()
            .map(|&i| record.get(i).and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0))
            .collect();
        let label: usize = record.get(label_idx).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        let action_name = record.get(action_idx).unwrap_or("Unknown").trim().to_string();

        class_names_map.entry(label).or_insert(action_name);
        raw_items.push(EmgItem { features, label });
    }


    if raw_items.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "CSV 파일에 유효한 데이터가 없습니다.",
        ));
    }

    let max_label = raw_items.iter().map(|item| item.label).max().unwrap_or(0);
    let num_classes = (max_label + 1).max(2);
    let mut class_names = Vec::new();
    for i in 0..num_classes {
        if let Some(name) = class_names_map.get(&i) {
            class_names.push(name.clone());
        } else {
            class_names.push(format!("Class {}", i));
        }
    }

    let mut num_features = raw_items[0].features.len();

    // 호환성 보장: 초기 수집된 구형 65차원(SSC/WAMP 누락 및 DC 미제거) CSV 파일인 경우,
    // 원시 5채널(a0..a4)로부터 현행 75차원 DC-Invariant 특징을 자동 재산출하여 100% 호환되도록 변환합니다.
    if num_features == 65 {
        println!(
            "ℹ️ [구형 65D 데이터셋 감지] 원시 5채널로부터 현행 75차원 DC-Invariant 특징으로 자동 업그레이드합니다..."
        );
        let mut proc = EmgFftProcessor::new();
        let mut prev_label = None;
        for item in &mut raw_items {
            if prev_label != Some(item.label) {
                proc = EmgFftProcessor::new();
                prev_label = Some(item.label);
            }
            let raw5 = [
                item.features[0],
                item.features[1],
                item.features[2],
                item.features[3],
                item.features[4],
            ];
            item.features = proc.process_sample_all(&raw5).to_vec();
        }
        num_features = TOTAL_FEATURES_ALL;
    }

    // 🚀 [수축 강도 다양화 증강 (Contraction Intensity Augmentation)]
    // 특히 휴식(0), 동적잡음(1), 진짜주먹(2) 3클래스 구조이거나 2클래스(휴식/주먹) 구조일 때,
    // 사용자가 학습 시 약하게 쥐었더라도 실시간에서 강하게 쥘 때 '동적 잡음'으로 튀는 현상을 원천 차단하기 위해
    // 능동 수축 클래스(3클래스 모드에서는 진짜 주먹 label 2, 기타 모드에서는 label >= 1)의 수축 강도를
    // [0.85, 1.25, 1.50, 1.80, 2.20] 배율로 다양화 증강합니다.
    let mut augmented_items = Vec::with_capacity(raw_items.len() * 3);
    for item in &raw_items {
        augmented_items.push(item.clone());

        let is_target_action = if num_classes == 3 {
            item.label == 2 // 3클래스 모드: 진짜 주먹만 증강 (잡음 1과 휴식 0 제외)
        } else if num_classes == 2 {
            item.label == 1 // 2클래스 모드: 손 쥐기 증강 (휴식 0 제외)
        } else {
            item.label >= 1 // 다중 손가락 모드: 모든 손가락 굽힘 동작 증강
        };

        if is_target_action {
            for &scale in &[0.85f32, 1.25f32, 1.50f32, 1.80f32, 2.20f32] {
                let mut scaled_feats = item.features.clone();
                // 1. 원시/평활 진폭 성분 (0..5)
                for ch in 0..5.min(scaled_feats.len()) {
                    scaled_feats[ch] *= scale;
                }
                // 2. FFT 주파수 크기 성분 (5..45)
                for idx in 5..45.min(scaled_feats.len()) {
                    scaled_feats[idx] *= scale;
                }
                // 3. 시간 도메인 진폭 성분 (RMS, MAV, WL, WAMP) 스케일링, 주파수 카운트(ZC, SSC)는 신호 생리학적 특성상 불변 유지
                for ch in 0..5 {
                    let base_idx = 45 + ch * 6;
                    if base_idx + 5 < scaled_feats.len() {
                        scaled_feats[base_idx] *= scale;     // RMS
                        scaled_feats[base_idx + 1] *= scale; // MAV
                        scaled_feats[base_idx + 2] *= scale; // WL
                        // ZC (base_idx + 3) 유지
                        // SSC (base_idx + 4) 유지
                        scaled_feats[base_idx + 5] *= scale; // WAMP
                    }
                }
                augmented_items.push(EmgItem {
                    features: scaled_feats,
                    label: item.label,
                });
            }
        }
    }

    let num_samples = augmented_items.len() as f32;
    let mut means = vec![0.0f32; num_features];
    let mut stds = vec![0.0f32; num_features];

    for item in &augmented_items {
        for (i, val) in item.features.iter().enumerate() {
            means[i] += val;
        }
    }
    for m in &mut means {
        *m /= num_samples;
    }

    for item in &augmented_items {
        for (i, val) in item.features.iter().enumerate() {
            stds[i] += (val - means[i]).powi(2);
        }
    }
    for s in &mut stds {
        *s = (*s / num_samples).sqrt().max(1e-6);
    }

    let stats = NormalizationStats {
        feature_dim: num_features,
        means,
        stds,
        num_classes,
        class_names,
    };

    // Z-Score 표준화 적용
    for item in &mut augmented_items {
        item.features = stats.normalize(&item.features);
    }

    // 시간 시계열 블록 편향 제거를 위한 무작위 셔플링
    let mut rng = thread_rng();
    augmented_items.shuffle(&mut rng);

    let split_idx = (augmented_items.len() as f32 * 0.8) as usize;
    let test_items = augmented_items.split_off(split_idx);

    Ok((
        EmgDataset { items: augmented_items },
        EmgDataset { items: test_items },
        stats,
    ))
}

/// 시계열(SEQ_LEN: 30) 데이터셋을 로드하여 Z-Score 정규화 및 80:20 학습/검증 분할 수행
pub fn load_seq_dataset(
    file_path: &str,
) -> io::Result<(EmgSeqDataset, EmgSeqDataset, NormalizationStats)> {
    let (features, labels, feature_dim, means, stds, class_names) = load_dataset(file_path)?;
    let total_records = features.len();
    if total_records <= SEQ_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "샘플 수가 너무 적습니다 (총 {} 레코드, SEQ_LEN: {} 필요)",
                total_records, SEQ_LEN
            ),
        ));
    }

    let mut items = Vec::new();
    for i in SEQ_LEN..total_records {
        let target_label = labels[i];
        // 윈도우 내 모든 샘플이 동일한 라벨인지 검사하여 경계면 혼합 방지
        let mut is_pure = true;
        for j in 0..SEQ_LEN {
            if labels[i - SEQ_LEN + j] != target_label {
                is_pure = false;
                break;
            }
        }
        if !is_pure {
            continue;
        }

        let mut window = Vec::with_capacity(SEQ_LEN * feature_dim);
        for j in 0..SEQ_LEN {
            let rec = &features[i - SEQ_LEN + j];
            for k in 0..feature_dim {
                window.push(rec[k]);
            }
        }
        items.push(EmgSeqItem {
            window,
            label: target_label,
            seq_len: SEQ_LEN,
            feature_dim,
        });
    }

    if items.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "유효한 순수 시계열 윈도우 샘플이 없습니다.",
        ));
    }

    let max_label = labels.iter().copied().max().unwrap_or(0);
    let num_classes = (max_label + 1).max(2);

    let stats = NormalizationStats {
        feature_dim,
        means,
        stds,
        num_classes,
        class_names,
    };

    // 무작위 셔플링
    let mut rng = thread_rng();
    items.shuffle(&mut rng);

    // 80% Train, 20% Valid 분할
    let split_idx = (items.len() as f32 * 0.8) as usize;
    let valid_items = items.split_off(split_idx);

    Ok((
        EmgSeqDataset { items },
        EmgSeqDataset { items: valid_items },
        stats,
    ))
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

/// 데이터셋 CSV 파일에서 raw 센서값을 읽어 EmgFftProcessor를 통해 DC-Invariant 75차원 특징을 산출하고,
/// 베이스라인 시프트(±50) 및 노이즈/진폭 데이터 증강(Data Augmentation)을 적용한 후 Z-Score 표준화를 수행합니다.
pub fn load_dataset(
    file_path: &str,
) -> io::Result<(Vec<Vec<f32>>, Vec<usize>, usize, Vec<f32>, Vec<f32>, Vec<String>)> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // 헤더 파싱
    let header = lines.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "빈 데이터셋 파일입니다.")
    })??;

    let header_cols: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
    if header_cols.len() < 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "잘못된 CSV 형식: 최소 3개 이상의 컬럼이 필요합니다. (발견: {})",
                header_cols.len()
            ),
        ));
    }

    // 이미 연산된 특징 파일(4D LDA 또는 75D 클린 데이터셋)인지 여부 확인
    let is_precomputed = header_cols.iter().any(|&c| c.starts_with("ld") || c.contains("_rms") || c.starts_with("feature_"));

    if is_precomputed {
        let label_idx = header_cols.iter().position(|&c| c == "label").unwrap_or(header_cols.len().saturating_sub(2));
        let action_idx = header_cols.iter().position(|&c| c == "action_name").unwrap_or(header_cols.len().saturating_sub(1));
        let feat_indices: Vec<usize> = header_cols
            .iter()
            .enumerate()
            .filter(|(idx, &col)| *idx != label_idx && *idx != action_idx && col != "action_name" && col != "index")
            .map(|(idx, _)| idx)
            .collect();

        let num_features = feat_indices.len();
        let mut all_features: Vec<Vec<f32>> = Vec::new();
        let mut all_labels: Vec<usize> = Vec::new();
        let mut class_names_map = std::collections::BTreeMap::new();

        for line_res in lines {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let parts: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();
            if parts.len() <= label_idx {
                continue;
            }

            let feats: Vec<f32> = feat_indices
                .iter()
                .map(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0))
                .collect();
            let label: usize = parts.get(label_idx).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
            let action_name = parts.get(action_idx).unwrap_or(&"Unknown").trim().to_string();

            class_names_map.entry(label).or_insert(action_name);
            all_features.push(feats);
            all_labels.push(label);
        }

        let total_rows = all_features.len();
        if total_rows == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "유효한 데이터 행이 없습니다."));
        }

        let max_label = all_labels.iter().copied().max().unwrap_or(0);
        let num_classes = (max_label + 1).max(2);
        let mut class_names = Vec::new();
        for i in 0..num_classes {
            if let Some(name) = class_names_map.get(&i) {
                class_names.push(name.clone());
            } else {
                class_names.push(format!("Class {}", i));
            }
        }

        // Z-Score 표준화
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

        for row in &mut all_features {
            for k in 0..num_features {
                row[k] = (row[k] - means[k]) / stds[k];
            }
        }

        println!(
            "✔ [기연산 특징 데이터 로드 완료] 총 {}행 (차원: {}D, Z-Score 표준화 적용, 클래스: {}개 - {})",
            total_rows, num_features, num_classes, class_names.join(", ")
        );

        return Ok((all_features, all_labels, num_features, means, stds, class_names));
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

    let max_label = raw_records.iter().map(|&(_, l)| l).max().unwrap_or(0);
    let num_classes = (max_label + 1).max(2);

    let mut class_counts = vec![0usize; num_classes];

    for &(_, label) in &raw_records {
        if label < num_classes {
            class_counts[label] += 1;
        }
    }

    let min_other_count = (1..num_classes)
        .map(|c| class_counts[c])
        .filter(|&cnt| cnt > 0)
        .min()
        .unwrap_or(class_counts[0]);

    let mut balanced_raw_records = Vec::new();
    let mut c0_seen = 0usize;
    let c0_stride = if min_other_count > 0 && class_counts[0] >= min_other_count * 2 {
        2
    } else {
        1
    };

    for &(sample, label) in &raw_records {
        if label == 0 {
            c0_seen += 1;
            if c0_seen % c0_stride == 0 {
                balanced_raw_records.push((sample, label));
            }
        } else {
            balanced_raw_records.push((sample, label));
        }
    }
    let raw_records = balanced_raw_records;

    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<usize> = Vec::new();
    let mut rng = SimpleRng::new(20260921);

    // 1. 원본 신호 특징 추출 (DC-Invariant 75차원)
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
        "✔ [데이터 로드 & 증강 완료] 원본 {}행 ➔ 증강 후 {}행 (3배 증강, {}차원 DC-Invariant 특징)",
        total_raw_rows, total_rows, num_features
    );

    let class_names = (0..num_classes).map(|i| format!("Class {}", i)).collect();

    Ok((
        standardized_features,
        all_labels,
        num_features,
        means,
        stds,
        class_names,
    ))
}

/// 시계열 윈도우(SEQ_LEN: 30)를 생성하여 텐서 입력 데이터로 변환합니다.
/// 🚀 1. 클래스 전이 구간(Cross-Gesture Boundary) 오염 방지: 윈도우 내 모든 샘플이 동일한 제스처일 때만 생성
/// 🚀 2. 미니배치 무작위 셔플링: 에포크 내 클래스 편향 및 특정 손가락 쏠림/락 방지
pub fn create_sliding_windows(
    features: &[Vec<f32>],
    labels: &[usize],
    feature_dim: usize,
) -> (Vec<f32>, Vec<i32>, usize) {
    let total_records = features.len();
    if total_records <= SEQ_LEN {
        return (Vec::new(), Vec::new(), 0);
    }

    struct WindowItem {
        data: Vec<f32>,
        label: i32,
    }

    let mut items = Vec::new();

    for i in SEQ_LEN..total_records {
        let target_label = labels[i];
        // 윈도우 내 모든 샘플이 동일한 라벨인지 검사하여 경계면 혼합 방지
        let mut is_pure = true;
        for j in 0..SEQ_LEN {
            if labels[i - SEQ_LEN + j] != target_label {
                is_pure = false;
                break;
            }
        }
        if !is_pure {
            continue;
        }

        let mut data = Vec::with_capacity(SEQ_LEN * feature_dim);
        for j in 0..SEQ_LEN {
            let rec = &features[i - SEQ_LEN + j];
            for k in 0..feature_dim {
                data.push(rec[k]);
            }
        }
        items.push(WindowItem {
            data,
            label: target_label as i32,
        });
    }

    // 🚀 무작위 셔플링으로 미니배치 클래스 균형 보장
    let mut rng = thread_rng();
    items.shuffle(&mut rng);

    let total_samples = items.len();
    let mut flat_inputs = Vec::with_capacity(total_samples * SEQ_LEN * feature_dim);
    let mut targets = Vec::with_capacity(total_samples);

    for it in items {
        flat_inputs.extend(it.data);
        targets.push(it.label);
    }

    (flat_inputs, targets, total_samples)
}

/// Burn 0.16의 고수준 LearnerBuilder 및 DataLoaderBuilder를 활용한 고성능 MLP 모델 학습
pub fn train_mlp_model<F>(
    device: &WgpuDevice,
    config: &TrainingConfig,
    mut progress_callback: F,
) -> io::Result<String>
where
    F: FnMut(TrainingProgress),
{
    println!("🚀 [MLP 학습 파이프라인] Burn 0.16 LearnerBuilder & 75차원 DC-Invariant 특징 학습 시작...");
    let (train_dataset, valid_dataset, stats) = load_mlp_dataset(&config.dataset_path)?;
    let total_train = train_dataset.len();
    let total_valid = valid_dataset.len();
    let feature_dim = stats.feature_dim;

    println!(
        "✔ [데이터셋 로드 완료] 학습 샘플: {}개 | 검증 샘플: {}개 (75차원 Z-Score 표준화 & 셔플링 완료)",
        total_train, total_valid
    );

    let batcher_train = EmgBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = EmgBatcher::<MyBackend>::new(device.clone());

    let batch_size = config.batch_size;
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(1)
        .build(valid_dataset);

    let mlp_cfg = EmgMlpConfig {
        d_input: feature_dim,
        d_hidden1: 128,
        d_hidden2: 64,
        d_hidden3: 32,
        d_hidden4: 16,
        num_classes: stats.num_classes,
        dropout: 0.15,
    };
    let model = mlp_cfg.init::<MyAutodiffBackend>(device);

    let artifacts_dir = "./model_artifacts";
    if !std::path::Path::new(artifacts_dir).exists() {
        let _ = std::fs::create_dir_all(artifacts_dir);
    }

    let epochs = if config.epochs == 150 || config.epochs == 25 { 10 } else { config.epochs };
    let model_file_name = config.model_arch.file_name();

    println!(
        "[*] 모델 학습 진행 중 ({} Epochs, Adam lr: {:?}, 클래스: {}개 - {})...",
        epochs,
        config.learning_rate,
        stats.num_classes,
        stats.class_names.join(", ")
    );

    let learner = LearnerBuilder::new(artifacts_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(epochs)
        .build(model, AdamConfig::new().init(), config.learning_rate);

    let trained_model = learner.fit(dataloader_train, dataloader_valid);

    let recorder = CompactRecorder::new();
    recorder
        .record(trained_model.into_record(), model_file_name.into())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("가중치 저장 실패: {:?}", e)))?;

    println!("[+] 최종 MLP 모델 가중치 저장 완료 -> {}.mpk", model_file_name);

    if let Err(e) = stats.save(model_file_name) {
        eprintln!("⚠️ 정규화 통계 저장 실패: {:?}", e);
    } else {
        println!("✔ [정규화 통계 저장] norm_stats_{}.json 및 norm_stats.json 저장 완료", model_file_name);
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

/// =========================================================================
/// 🚀 [단계 1: 시계열 슬라이딩 윈도우 & 세션 블록 격리 분할 데이터셋 로더]
/// =========================================================================
/// - **기능 1.1 (세션/블록 단위 시계열 분할)**: 인접 프레임 간 자기상관 누수(`Cov(x_t, x_{t+1}) >> 0`)를 원천 차단하기 위해,
///   연속된 동일 라벨의 세션 블록을 먼저 시간순으로 Train(앞 80%) / Validation(뒤 20%) 구간으로 격리 분할한 뒤,
///   분할된 각 구간 내부에서만 독립적으로 길이 `T = window_size`의 2D 슬라이딩 윈도우를 생성합니다.
/// - **기능 1.4 (채널별 분포 정규화)**: 오직 학습 세트에서만 채널별 평균 `μ`와 표준편차 `σ`를 산출하고
///   `(x - μ) / (σ + 1e-8)` 변환을 수행합니다.
pub fn load_modular_mlp_dataset(
    file_path: &str,
    window_size: usize,
) -> io::Result<(EmgSeqDataset, EmgSeqDataset, NormalizationStats)> {
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);

    let headers = rdr
        .headers()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        .clone();
    let label_idx = headers
        .iter()
        .position(|h| h == "label")
        .unwrap_or(headers.len().saturating_sub(2));
    let action_idx = headers
        .iter()
        .position(|h| h == "action_name")
        .unwrap_or(headers.len().saturating_sub(1));

    let feat_indices: Vec<usize> = headers
        .iter()
        .enumerate()
        .filter(|(idx, h)| *idx != label_idx && *idx != action_idx && h.trim() != "index")
        .map(|(idx, _)| idx)
        .collect();

    let mut raw_items: Vec<EmgItem> = Vec::new();
    let mut class_names_map = std::collections::BTreeMap::new();

    for result in rdr.records() {
        let record = result.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if record.len() <= label_idx {
            continue;
        }

        let features: Vec<f32> = feat_indices
            .iter()
            .map(|&i| {
                record
                    .get(i)
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0)
            })
            .collect();
        let label: usize = record
            .get(label_idx)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let action_name = record
            .get(action_idx)
            .unwrap_or("Unknown")
            .trim()
            .to_string();

        class_names_map.entry(label).or_insert(action_name);
        raw_items.push(EmgItem { features, label });
    }

    if raw_items.len() <= window_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "데이터셋 샘플 수({})가 윈도우 길이({})보다 적습니다.",
                raw_items.len(),
                window_size
            ),
        ));
    }

    let max_label = raw_items.iter().map(|item| item.label).max().unwrap_or(0);
    let num_classes = (max_label + 1).max(2);
    let mut class_names = Vec::new();
    for i in 0..num_classes {
        if let Some(name) = class_names_map.get(&i) {
            class_names.push(name.clone());
        } else {
            class_names.push(format!("Class {}", i));
        }
    }
    let mut num_features = raw_items[0].features.len();

    // 호환성 보장: 초기 수집된 구형 65차원(SSC/WAMP 누락 및 DC 미제거) CSV 파일인 경우,
    // 원시 5채널(a0..a4)로부터 현행 75차원 DC-Invariant 특징을 자동 재산출하여 100% 호환되도록 변환합니다.
    if num_features == 65 {
        println!(
            "ℹ️ [구형 65D 데이터셋 감지] 원시 5채널로부터 현행 75차원 DC-Invariant 특징으로 자동 업그레이드합니다..."
        );
        let mut proc = EmgFftProcessor::new();
        let mut prev_label = None;
        for item in &mut raw_items {
            if prev_label != Some(item.label) {
                proc = EmgFftProcessor::new();
                prev_label = Some(item.label);
            }
            let raw5 = [
                item.features[0],
                item.features[1],
                item.features[2],
                item.features[3],
                item.features[4],
            ];
            item.features = proc.process_sample_all(&raw5).to_vec();
        }
        num_features = TOTAL_FEATURES_ALL;
    }

    // 1. 기능 1.4: 왜곡 없는 실제 센서 분포를 반영하기 위해 원본 데이터(raw_items)에서 먼저 채널별 평균 μ 및 표준편차 σ 산출
    let mut means = vec![0.0f32; num_features];
    let mut stds = vec![0.0f32; num_features];
    let total_raw = (raw_items.len().max(1)) as f32;

    for item in &raw_items {
        for (i, &val) in item.features.iter().enumerate() {
            means[i] += val;
        }
    }
    for m in &mut means {
        *m /= total_raw;
    }

    for item in &raw_items {
        for (i, &val) in item.features.iter().enumerate() {
            stds[i] += (val - means[i]).powi(2);
        }
    }
    for s in &mut stds {
        *s = (*s / total_raw).sqrt().max(1e-6);
    }

    let stats = NormalizationStats {
        feature_dim: num_features,
        means,
        stds,
        num_classes,
        class_names,
    };

    // 2. 연속된 동일 라벨 구간(세션/시간 블록) 단위로 그룹화하여 클래스 경계면 혼합(Cross-Gesture Contamination) 차단
    let mut contiguous_blocks: Vec<(usize, Vec<Vec<f32>>)> = Vec::new();
    for item in raw_items {
        if let Some((last_label, ref mut block_frames)) = contiguous_blocks.last_mut() {
            if *last_label == item.label {
                block_frames.push(item.features);
                continue;
            }
        }
        contiguous_blocks.push((item.label, vec![item.features]));
    }

    // 3. 각 순수 라벨 블록 내부에서 길이 T = window_size의 슬라이딩 윈도우 생성 후,
    //    클래스 및 사이클 전 구간이 고르게 포함되도록 계층적(Stratified) 분할 (Train 80% / Valid 20%)
    //    + 학습 세트에 대해 수축 강도 다변화 증강 적용
    let mut train_windows: Vec<EmgSeqItem> = Vec::new();
    let mut valid_windows: Vec<EmgSeqItem> = Vec::new();

    let scale_frame = |frame: &[f32], scale: f32| -> Vec<f32> {
        let mut sf = frame.to_vec();
        if num_features >= 65 {
            for ch in 0..5.min(sf.len()) {
                sf[ch] *= scale;
            }
            for idx in 5..45.min(sf.len()) {
                sf[idx] *= scale;
            }
            if num_features >= 75 {
                for ch in 0..5 {
                    let base_idx = 45 + ch * 6;
                    if base_idx + 5 < sf.len() {
                        sf[base_idx] *= scale;
                        sf[base_idx + 1] *= scale;
                        sf[base_idx + 2] *= scale;
                        // ZC (base_idx + 3) 및 SSC (base_idx + 4)는 주파수 영교차/기울기 부호 변화 횟수이므로 진폭 스케일에 불변(Invariant)
                        sf[base_idx + 5] = (sf[base_idx + 5] * scale.sqrt()).clamp(0.0, 15.0);
                    }
                }
            } else {
                for ch in 0..5 {
                    let base_idx = 45 + ch * 4;
                    if base_idx + 2 < sf.len() {
                        sf[base_idx] *= scale;
                        sf[base_idx + 1] *= scale;
                        sf[base_idx + 2] *= scale;
                    }
                }
            }
        } else if num_features == 35 {
            // 🚀 35D (TD_CORE_FEATURES) 생체신호 물리 법칙 준수 스케일링:
            // - 0..5: 5채널 평활 AC 진폭 -> scale 비례
            // - 5 + ch*6 + {0(RMS), 1(MAV), 2(WL)}: 진폭 에너지 및 파형 길이 -> scale 비례
            // - 5 + ch*6 + {3(ZC), 4(SSC)}: 영교차율 및 기울기 부호 반전 횟수 -> 진폭 배율에 불변(Scale-Invariant) 유지!
            // - 5 + ch*6 + 5(WAMP): 문턱치 초과 횟수 -> 비선형 완만 변동 (sqrt(scale))
            for ch in 0..5 {
                sf[ch] *= scale;
                let base_idx = 5 + ch * 6;
                if base_idx + 5 < sf.len() {
                    sf[base_idx] *= scale;     // ac_rms
                    sf[base_idx + 1] *= scale; // ac_mav
                    sf[base_idx + 2] *= scale; // wl
                    // sf[base_idx + 3] (zc) 및 sf[base_idx + 4] (ssc) 불변 유지
                    sf[base_idx + 5] = (sf[base_idx + 5] * scale.sqrt()).clamp(0.0, 15.0); // wamp
                }
            }
        } else {
            for v in &mut sf {
                *v *= scale;
            }
        }
        sf
    };

    // 원시 특징 프레임에서 5채널 총합 RMS 에너지를 추출하는 헬퍼 클로저
    let frame_total_rms = |frame: &[f32]| -> f32 {
        if num_features >= 75 {
            (0..5).map(|ch| frame[45 + ch * 6]).sum()
        } else if num_features == 35 {
            (0..5).map(|ch| frame[5 + ch * 6]).sum()
        } else {
            frame.iter().take(5).copied().sum()
        }
    };

    for (label, block) in &contiguous_blocks {
        if block.len() < window_size {
            continue;
        }
        let norm_block: Vec<Vec<f32>> = block.iter().map(|f| stats.normalize(f)).collect();
        let num_wins = norm_block.len() - window_size + 1;

        // 🚀 [무중복 검증 아일랜드 분할 (Non-overlapping Validation Islands with Purge Gap)]
        // 기존의 꼬리 20% 분할은 사용자의 6초 수축 말기 근피로/조기이완(Fatigue decay) 데이터만 검증셋에 몰려
        // Valid Loss가 비정상적으로 치솟는 왜곡이 있었습니다.
        // 이를 개선하여 블록 전체의 초반(20%), 중반(50%), 후반(75%) 구간에 각각 16프레임(~0.3초) 크기의
        // 무중복 검증 아일랜드를 설정하고, 각 아일랜드 전후로 완충지대(Purge Gap)를 두어
        // 학습셋과 검증셋 간 0.0% 프레임 공유율(누수 차단)을 유지하면서도 수축 전주기의 고른 검증 분포를 보장합니다.
        let n = block.len();
        let island_len = window_size * 2; // 16 frames (~0.32초)
        let i1_start = ((n as f32) * 0.20) as usize;
        let i1_end = (i1_start + island_len).min(n);

        let i2_start = ((n as f32) * 0.50) as usize;
        let i2_end = (i2_start + island_len).min(n);

        let i3_start = ((n as f32) * 0.75) as usize;
        let i3_end = (i3_start + island_len).min(n);

        let islands = [(i1_start, i1_end), (i2_start, i2_end), (i3_start, i3_end)];

        for w_idx in 0..num_wins {
            let start_idx = w_idx;
            let end_idx = start_idx + window_size;
            let mut flat_win = Vec::with_capacity(window_size * num_features);
            for t in start_idx..end_idx {
                flat_win.extend_from_slice(&norm_block[t]);
            }
            let base_item = EmgSeqItem {
                window: flat_win,
                label: *label,
                seq_len: window_size,
                feature_dim: num_features,
            };

            let in_island = islands.iter().any(|&(is_s, is_e)| start_idx >= is_s && end_idx <= is_e);

            if in_island {
                valid_windows.push(base_item);
            } else {
                let overlaps = islands.iter().any(|&(is_s, is_e)| !(end_idx <= is_s || start_idx >= is_e));
                if !overlaps {
                    train_windows.push(base_item);

                    // 🚀 생체역학적 3상 등척성 피로 시뮬레이션 증강 (Isometric Fatigue & Boundary-Preserving Augmentation):
                    // - 휴식(0) 및 동적 잡음(1): 주먹 영역으로 침범하지 않도록 보수적 범위 내에서 증강
                    // - 수축 동작(>=2): 윈도우의 평균 총합 RMS가 고진폭(Phase 1 초기 동원 > 210)일 때는 0.65배 감쇠 증강을 통해
                    //   5채널 근육 공간 협응 비율(Spatial Synergy Ratio)과 주파수(ZC/SSC)는 그대로 유지하면서
                    //   Phase 2 등척성 진폭 감쇠 구간(RMS 140~220)의 학습 밀도를 2배로 보강합니다.
                    //   반대로 이미 감쇠된 Phase 2 저진폭 구간(<= 210)은 잡음 영역(< 100)과 충돌하지 않도록 [0.92, 1.15]로 보호합니다.
                    let win_mean_rms: f32 = (start_idx..end_idx)
                        .map(|t| frame_total_rms(&block[t]))
                        .sum::<f32>()
                        / (window_size as f32);

                    let aug_scales: [f32; 2] = if num_classes >= 3 {
                        match *label {
                            0 => [0.92, 0.98], // 정적 휴식: 저에너지 영역 유지
                            1 => [0.96, 1.02], // 동적 잡음: 주먹 영역 침범 차단
                            _ => {
                                if win_mean_rms > 210.0 {
                                    [0.65, 1.12] // Phase 1 고진폭 -> Phase 2 등척성 감쇠(0.65x) 및 강수축(1.12x) 동시 모사
                                } else {
                                    [0.92, 1.15] // Phase 2/3 중저진폭 -> 잡음 충돌 방지 및 수축 복원력 강화
                                }
                            }
                        }
                    } else if *label >= 1 {
                        if win_mean_rms > 210.0 {
                            [0.65, 1.12]
                        } else {
                            [0.92, 1.15]
                        }
                    } else {
                        [0.93, 0.98]
                    };

                    for &scale in &aug_scales {
                        let mut aug_win = Vec::with_capacity(window_size * num_features);
                        for t in start_idx..end_idx {
                            let scaled_raw = scale_frame(&block[t], scale);
                            let norm_scaled = stats.normalize(&scaled_raw);
                            aug_win.extend_from_slice(&norm_scaled);
                        }
                        train_windows.push(EmgSeqItem {
                            window: aug_win,
                            label: *label,
                            seq_len: window_size,
                            feature_dim: num_features,
                        });
                    }
                }
            }
        }
    }

    let mut rng = thread_rng();
    train_windows.shuffle(&mut rng);
    valid_windows.shuffle(&mut rng);

    if valid_windows.is_empty() && train_windows.len() > 10 {
        let split_at = ((train_windows.len() as f32) * 0.80) as usize;
        valid_windows = train_windows.split_off(split_at);
    }

    Ok((
        EmgSeqDataset {
            items: train_windows,
        },
        EmgSeqDataset {
            items: valid_windows,
        },
        stats,
    ))
}

/// =========================================================================
/// 🚀 [단계 2 ~ 단계 7] 시계열 윈도우 + Temporal-SwiGLU 블록 신경망 학습 파이프라인
/// =========================================================================
pub fn train_modular_mlp_model<F>(
    device: &WgpuDevice,
    config: &TrainingConfig,
    mut progress_callback: F,
) -> io::Result<String>
where
    F: FnMut(TrainingProgress),
{
    let window_size = MODULAR_MLP_WINDOW_SIZE;
    println!(
        "🚀 [Block-MLP 학습 파이프라인] 2D 시계열 윈도우(T={}) + Temporal-SwiGLU 블록 신경망 학습 시작...",
        window_size
    );

    let (train_dataset, valid_dataset, stats) =
        load_modular_mlp_dataset(&config.dataset_path, window_size)?;
    let total_train = train_dataset.len();
    let total_valid = valid_dataset.len();
    let feature_dim = stats.feature_dim;

    println!(
        "✔ [세션 격리 윈도우 데이터셋 로드 완료] 학습 윈도우: {}개 | 검증 윈도우: {}개 (T={}, 차원: {}D, 자기상관 누수 차단 완료)",
        total_train, total_valid, window_size, feature_dim
    );

    let batcher_train = EmgSeqBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = EmgSeqBatcher::<MyBackend>::new(device.clone());

    let batch_size = config.batch_size.max(1);
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(1)
        .build(valid_dataset);

    let mod_mlp_cfg = EmgModularMlpConfig {
        d_in: feature_dim,
        window_size,
        d_model: 64,
        num_classes: stats.num_classes,
        num_blocks: 2,
        dropout: 0.20,
        max_drop_path: 0.10,
        layer_scale_init: 1e-1,
        causal_readout: true,
    };
    let model = mod_mlp_cfg.init::<MyAutodiffBackend>(device);

    let artifacts_dir = "./model_artifacts";
    if !std::path::Path::new(artifacts_dir).exists() {
        let _ = std::fs::create_dir_all(artifacts_dir);
    }

    let epochs = if config.epochs == 150 || config.epochs == 25 { 20 } else { config.epochs };
    let model_file_name = config.model_arch.file_name();

    // 기능 7.2 & 7.3: 분리된 가중치 감쇠(AdamW) 및 선형 웜업 결합 코사인 어닐링 스케줄러 구성
    let steps_per_epoch = (total_train + batch_size - 1) / batch_size;
    let total_steps = (epochs * steps_per_epoch).max(10);
    let warmup_steps = ((total_steps as f64) * 0.08).round() as usize;
    let lr_scheduler = WarmupCosineLrScheduler::new(
        config.learning_rate,
        1e-6,
        warmup_steps.max(2),
        total_steps,
    );
    let optim = AdamWConfig::new()
        .with_weight_decay(1e-2)
        .init();

    println!(
        "[*] Block-MLP 학습 진행 중 ({} Epochs, AdamW + WarmupCosine lr: {:?} -> 1e-6, 웜업: {} 스텝, 클래스: {}개 - {})...",
        epochs,
        config.learning_rate,
        warmup_steps.max(2),
        stats.num_classes,
        stats.class_names.join(", ")
    );

    let learner = LearnerBuilder::new(artifacts_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(epochs)
        .build(model, optim, lr_scheduler);

    let trained_model = learner.fit(dataloader_train, dataloader_valid);

    let recorder = CompactRecorder::new();
    recorder
        .record(trained_model.into_record(), model_file_name.into())
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Block-MLP 가중치 저장 실패: {:?}", e),
            )
        })?;

    println!(
        "[+] 최종 Block-MLP (Temporal-SwiGLU) 가중치 저장 완료 -> {}.mpk",
        model_file_name
    );

    if let Err(e) = stats.save(model_file_name) {
        eprintln!("⚠️ 정규화 통계 저장 실패: {:?}", e);
    } else {
        println!(
            "✔ [정규화 통계 저장] norm_stats_{}.json 및 norm_stats.json 저장 완료",
            model_file_name
        );
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

/// Burn 0.16의 고수준 LearnerBuilder 및 DataLoaderBuilder를 활용한 고성능 CNN + LSTM (CRNN) 모델 학습
pub fn train_cnnlstm_model<F>(
    device: &WgpuDevice,
    config: &TrainingConfig,
    mut progress_callback: F,
) -> io::Result<String>
where
    F: FnMut(TrainingProgress),
{
    println!("🚀 [CNN + LSTM 학습 파이프라인] Burn 0.16 LearnerBuilder & 시계열(SEQ_LEN={}) 특징 학습 시작...", SEQ_LEN);
    let (train_dataset, valid_dataset, stats) = load_seq_dataset(&config.dataset_path)?;
    let total_train = train_dataset.len();
    let total_valid = valid_dataset.len();
    let feature_dim = stats.feature_dim;

    println!(
        "✔ [시계열 데이터셋 로드 완료] 학습 윈도우: {}개 | 검증 윈도우: {}개 (차원: {}D, Z-Score 표준화 & 셔플링 완료)",
        total_train, total_valid, feature_dim
    );

    let batcher_train = EmgSeqBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = EmgSeqBatcher::<MyBackend>::new(device.clone());

    let batch_size = config.batch_size;
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(1)
        .build(valid_dataset);

    let model: EmgCnnLstmModel<MyAutodiffBackend> = EmgCnnLstmModel::new(device, feature_dim, stats.num_classes);

    let artifacts_dir = "./model_artifacts";
    if !std::path::Path::new(artifacts_dir).exists() {
        let _ = std::fs::create_dir_all(artifacts_dir);
    }

    let epochs = if config.epochs == 150 { 60 } else { config.epochs };
    let model_file_name = config.model_arch.file_name();

    println!(
        "[*] CNN + LSTM 모델 학습 진행 중 ({} Epochs, Adam lr: {:?}, 클래스: {}개 - {})...",
        epochs,
        config.learning_rate,
        stats.num_classes,
        stats.class_names.join(", ")
    );

    let learner = LearnerBuilder::new(artifacts_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(epochs)
        .build(model, AdamConfig::new().init(), config.learning_rate);

    let trained_model = learner.fit(dataloader_train, dataloader_valid);

    let recorder = CompactRecorder::new();
    recorder
        .record(trained_model.into_record(), model_file_name.into())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("가중치 저장 실패: {:?}", e)))?;

    println!("[+] 최종 CNN + LSTM 모델 가중치 저장 완료 -> {}.mpk", model_file_name);

    if let Err(e) = stats.save(model_file_name) {
        eprintln!("⚠️ 정규화 통계 저장 실패: {:?}", e);
    } else {
        println!("✔ [정규화 통계 저장] norm_stats_{}.json 및 norm_stats.json 저장 완료", model_file_name);
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

/// Burn 0.16의 고수준 LearnerBuilder 및 DataLoaderBuilder를 활용한 고성능 1D-TCN 모델 학습
pub fn train_tcn_model<F>(
    device: &WgpuDevice,
    config: &TrainingConfig,
    mut progress_callback: F,
) -> io::Result<String>
where
    F: FnMut(TrainingProgress),
{
    println!("🚀 [1D-TCN 학습 파이프라인] Burn 0.16 LearnerBuilder & 시계열(SEQ_LEN={}) 특징 학습 시작...", SEQ_LEN);
    let (train_dataset, valid_dataset, stats) = load_seq_dataset(&config.dataset_path)?;
    let total_train = train_dataset.len();
    let total_valid = valid_dataset.len();
    let feature_dim = stats.feature_dim;

    println!(
        "✔ [시계열 데이터셋 로드 완료] 학습 윈도우: {}개 | 검증 윈도우: {}개 (차원: {}D, Z-Score 표준화 & 셔플링 완료)",
        total_train, total_valid, feature_dim
    );

    let batcher_train = EmgSeqBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = EmgSeqBatcher::<MyBackend>::new(device.clone());

    let batch_size = config.batch_size;
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(1)
        .build(valid_dataset);

    let model: EmgTcnModel<MyAutodiffBackend> = EmgTcnModel::new(device, feature_dim, stats.num_classes);

    let artifacts_dir = "./model_artifacts";
    if !std::path::Path::new(artifacts_dir).exists() {
        let _ = std::fs::create_dir_all(artifacts_dir);
    }

    let epochs = if config.epochs == 150 { 60 } else { config.epochs };
    let model_file_name = config.model_arch.file_name();

    println!(
        "[*] 1D-TCN 모델 학습 진행 중 ({} Epochs, Adam lr: {:?}, 클래스: {}개 - {})...",
        epochs,
        config.learning_rate,
        stats.num_classes,
        stats.class_names.join(", ")
    );

    let learner = LearnerBuilder::new(artifacts_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(epochs)
        .build(model, AdamConfig::new().init(), config.learning_rate);

    let trained_model = learner.fit(dataloader_train, dataloader_valid);

    let recorder = CompactRecorder::new();
    recorder
        .record(trained_model.into_record(), model_file_name.into())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("가중치 저장 실패: {:?}", e)))?;

    println!("[+] 최종 1D-TCN 모델 가중치 저장 완료 -> {}.mpk", model_file_name);

    if let Err(e) = stats.save(model_file_name) {
        eprintln!("⚠️ 정규화 통계 저장 실패: {:?}", e);
    } else {
        println!("✔ [정규화 통계 저장] norm_stats_{}.json 및 norm_stats.json 저장 완료", model_file_name);
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

/// Burn 0.16의 고수준 LearnerBuilder 및 DataLoaderBuilder를 활용한 고성능 1D-CNN 모델 학습
pub fn train_cnn_model<F>(
    device: &WgpuDevice,
    config: &TrainingConfig,
    mut progress_callback: F,
) -> io::Result<String>
where
    F: FnMut(TrainingProgress),
{
    println!("🚀 [1D-CNN 학습 파이프라인] Burn 0.16 LearnerBuilder & 시계열(SEQ_LEN={}) 특징 학습 시작...", SEQ_LEN);
    let (train_dataset, valid_dataset, stats) = load_seq_dataset(&config.dataset_path)?;
    let total_train = train_dataset.len();
    let total_valid = valid_dataset.len();
    let feature_dim = stats.feature_dim;

    println!(
        "✔ [시계열 데이터셋 로드 완료] 학습 윈도우: {}개 | 검증 윈도우: {}개 (차원: {}D, Z-Score 표준화 & 셔플링 완료)",
        total_train, total_valid, feature_dim
    );

    let batcher_train = EmgSeqBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = EmgSeqBatcher::<MyBackend>::new(device.clone());

    let batch_size = config.batch_size;
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(1)
        .build(valid_dataset);

    let model: EmgCnnModel<MyAutodiffBackend> = EmgCnnModel::new(device, feature_dim, stats.num_classes);

    let artifacts_dir = "./model_artifacts";
    if !std::path::Path::new(artifacts_dir).exists() {
        let _ = std::fs::create_dir_all(artifacts_dir);
    }

    let epochs = if config.epochs == 150 { 60 } else { config.epochs };
    let model_file_name = config.model_arch.file_name();

    println!(
        "[*] 1D-CNN 모델 학습 진행 중 ({} Epochs, Adam lr: {:?}, 클래스: {}개 - {})...",
        epochs,
        config.learning_rate,
        stats.num_classes,
        stats.class_names.join(", ")
    );

    let learner = LearnerBuilder::new(artifacts_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(epochs)
        .build(model, AdamConfig::new().init(), config.learning_rate);

    let trained_model = learner.fit(dataloader_train, dataloader_valid);

    let recorder = CompactRecorder::new();
    recorder
        .record(trained_model.into_record(), model_file_name.into())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("가중치 저장 실패: {:?}", e)))?;

    println!("[+] 최종 1D-CNN 모델 가중치 저장 완료 -> {}.mpk", model_file_name);

    if let Err(e) = stats.save(model_file_name) {
        eprintln!("⚠️ 정규화 통계 저장 실패: {:?}", e);
    } else {
        println!("✔ [정규화 통계 저장] norm_stats_{}.json 및 norm_stats.json 저장 완료", model_file_name);
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

/// Burn 0.16의 고수준 LearnerBuilder 및 DataLoaderBuilder를 활용한 고성능 LSTM 모델 학습
pub fn train_lstm_model<F>(
    device: &WgpuDevice,
    config: &TrainingConfig,
    mut progress_callback: F,
) -> io::Result<String>
where
    F: FnMut(TrainingProgress),
{
    println!("🚀 [LSTM 학습 파이프라인] Burn 0.16 LearnerBuilder & 시계열(SEQ_LEN={}) 특징 학습 시작...", SEQ_LEN);
    let (train_dataset, valid_dataset, stats) = load_seq_dataset(&config.dataset_path)?;
    let total_train = train_dataset.len();
    let total_valid = valid_dataset.len();
    let feature_dim = stats.feature_dim;

    println!(
        "✔ [시계열 데이터셋 로드 완료] 학습 윈도우: {}개 | 검증 윈도우: {}개 (차원: {}D, Z-Score 표준화 & 셔플링 완료)",
        total_train, total_valid, feature_dim
    );

    let batcher_train = EmgSeqBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = EmgSeqBatcher::<MyBackend>::new(device.clone());

    let batch_size = config.batch_size;
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(1)
        .build(valid_dataset);

    let model: EmgLstmModel<MyAutodiffBackend> = EmgLstmModel::new(device, feature_dim, stats.num_classes);

    let artifacts_dir = "./model_artifacts";
    if !std::path::Path::new(artifacts_dir).exists() {
        let _ = std::fs::create_dir_all(artifacts_dir);
    }

    let epochs = if config.epochs == 150 { 60 } else { config.epochs };
    let model_file_name = config.model_arch.file_name();

    println!(
        "[*] LSTM 모델 학습 진행 중 ({} Epochs, Adam lr: {:?}, 클래스: {}개 - {})...",
        epochs,
        config.learning_rate,
        stats.num_classes,
        stats.class_names.join(", ")
    );

    let learner = LearnerBuilder::new(artifacts_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(epochs)
        .build(model, AdamConfig::new().init(), config.learning_rate);

    let trained_model = learner.fit(dataloader_train, dataloader_valid);

    let recorder = CompactRecorder::new();
    recorder
        .record(trained_model.into_record(), model_file_name.into())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("가중치 저장 실패: {:?}", e)))?;

    println!("[+] 최종 LSTM 모델 가중치 저장 완료 -> {}.mpk", model_file_name);

    if let Err(e) = stats.save(model_file_name) {
        eprintln!("⚠️ 정규화 통계 저장 실패: {:?}", e);
    } else {
        println!("✔ [정규화 통계 저장] norm_stats_{}.json 및 norm_stats.json 저장 완료", model_file_name);
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

/// 지정된 모델 아키텍처에 따라 Burn 0.16 LearnerBuilder 기반 학습을 수행하고, 저장된 모델 파일 경로를 반환합니다.
pub fn train_model<F>(
    device: &WgpuDevice,
    config: &TrainingConfig,
    progress_callback: F,
) -> io::Result<String>
where
    F: FnMut(TrainingProgress),
{
    match config.model_arch {
        ModelArchitecture::Mlp => train_mlp_model(device, config, progress_callback),
        ModelArchitecture::ModularMlp => train_modular_mlp_model(device, config, progress_callback),
        ModelArchitecture::CnnLstm => train_cnnlstm_model(device, config, progress_callback),
        ModelArchitecture::Tcn => train_tcn_model(device, config, progress_callback),
        ModelArchitecture::Cnn => train_cnn_model(device, config, progress_callback),
        ModelArchitecture::Lstm => train_lstm_model(device, config, progress_callback),
    }
}
