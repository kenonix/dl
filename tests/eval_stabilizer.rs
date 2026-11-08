use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::Tensor;
use emg_ml_pipeline::models::EmgMlpConfig;
use emg_ml_pipeline::train::{MyBackend, NormalizationStats};

pub struct GestureStabilizer {
    pub ema_probs: Vec<f32>,
    pub current_stable_gesture: usize,
    pub candidate_gesture: usize,
    pub candidate_count: usize,
    pub alpha: f32,
    pub enter_threshold: f32,
    pub hold_threshold: f32,
    pub debounce_frames: usize,
    pub is_energy_at_rest: bool,
    pub rest_enter_rms: f32,
    pub rest_exit_rms: f32,
    pub rest_debounce_count: usize,
}

impl GestureStabilizer {
    pub fn new(num_classes: usize, base_rest_rms: f32) -> Self {
        let mut init_probs = vec![0.0f32; num_classes];
        if !init_probs.is_empty() {
            init_probs[0] = 1.0;
        }
        let enter_thresh = (1.25 / num_classes as f32).clamp(0.35, 0.48);
        let hold_thresh = (0.85 / num_classes as f32).clamp(0.22, 0.32);

        let enter_rms = (base_rest_rms * 1.15).max(20.0);
        let exit_rms = (base_rest_rms * 1.45).max(enter_rms + 6.0);

        Self {
            ema_probs: init_probs,
            current_stable_gesture: 0,
            candidate_gesture: 0,
            candidate_count: 0,
            alpha: 0.30,
            enter_threshold: enter_thresh,
            hold_threshold: hold_thresh,
            debounce_frames: 3,
            is_energy_at_rest: true,
            rest_enter_rms: enter_rms,
            rest_exit_rms: exit_rms,
            rest_debounce_count: 0,
        }
    }

    pub fn update(&mut self, raw_probs: &[f32], total_ac_rms: f32) -> (usize, f32, &'static str) {
        let num_classes = raw_probs.len();
        if self.ema_probs.len() != num_classes {
            self.ema_probs = vec![0.0; num_classes];
            self.ema_probs[0] = 1.0;
        }

        for i in 0..num_classes {
            self.ema_probs[i] = self.alpha * raw_probs[i] + (1.0 - self.alpha) * self.ema_probs[i];
        }

        // 2. EMA 기반 최적 후보 선정
        let mut best_cand = 0;
        let mut best_p = self.ema_probs[0];
        for (i, &p) in self.ema_probs.iter().enumerate().skip(1) {
            if p > best_p {
                best_p = p;
                best_cand = i;
            }
        }

        let mut state_tag;
        if best_cand == self.current_stable_gesture {
            self.candidate_gesture = best_cand;
            self.candidate_count = 0;
            state_tag = "안정";
        } else {
            let current_hold_p = self.ema_probs.get(self.current_stable_gesture).copied().unwrap_or(0.0);
            
            let current_collapsed = current_hold_p < self.hold_threshold;
            let strong_candidate = best_p >= self.enter_threshold && (best_p > current_hold_p * 1.15 || current_collapsed);
            let rest_recovery = best_cand == 0 && (best_p > current_hold_p || total_ac_rms < self.rest_enter_rms * 1.15);

            if current_collapsed || strong_candidate || rest_recovery {
                if best_cand == self.candidate_gesture {
                    self.candidate_count += 1;
                } else {
                    self.candidate_gesture = best_cand;
                    self.candidate_count = 1;
                }
                state_tag = "전이";

                let needed_frames = if current_collapsed || rest_recovery { 2 } else { self.debounce_frames };
                if self.candidate_count >= needed_frames {
                    self.current_stable_gesture = best_cand;
                    self.candidate_count = 0;
                    state_tag = "확정";
                }
            } else {
                self.candidate_count = 0;
                state_tag = "유지";
            }
        }

        let stable_p = self.ema_probs.get(self.current_stable_gesture).copied().unwrap_or(0.0);
        (self.current_stable_gesture, stable_p, state_tag)
    }
}

#[test]
fn test_stabilizer_with_dataset() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let dataset_path = "emg_dataset_noise_robust_20261002_152455.csv";
    if !std::path::Path::new(dataset_path).exists() {
        return;
    }

    let stats = NormalizationStats::load_for_model("emg_mlp_model").expect("Failed to load norm stats");
    let num_classes = stats.num_classes;

    let mut model = EmgMlpConfig {
        d_input: stats.feature_dim,
        d_hidden1: 128,
        d_hidden2: 64,
        d_hidden3: 32,
        d_hidden4: 16,
        num_classes,
        dropout: 0.15,
    }
    .init::<MyBackend>(&device);

    let recorder = CompactRecorder::new();
    let rec = recorder.load("emg_mlp_model".into(), &device).expect("Failed to load weights");
    model = model.load_record(rec);

    let file = std::fs::File::open(dataset_path).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let headers = rdr.headers().unwrap().clone();
    let label_idx = headers.iter().position(|h| h == "label").unwrap();
    let rms_indices: Vec<usize> = headers
        .iter()
        .enumerate()
        .filter(|(_, h)| h.ends_with("_rms"))
        .map(|(i, _)| i)
        .collect();

    let mut stabilizer = GestureStabilizer::new(num_classes, 48.64);
    let mut confusion: Vec<Vec<usize>> = vec![vec![0; num_classes]; num_classes];

    for res in rdr.records() {
        let rec = res.unwrap();
        let label: usize = rec[label_idx].parse().unwrap();
        let feats: Vec<f32> = (0..stats.feature_dim).map(|i| rec[i].parse().unwrap_or(0.0)).collect();
        let total_ac_rms: f32 = rms_indices.iter().map(|&idx| rec[idx].parse::<f32>().unwrap_or(0.0)).sum();

        let norm_feats = stats.normalize(&feats);
        let input_tensor = Tensor::<MyBackend, 1>::from_floats(norm_feats.as_slice(), &device)
            .reshape([1, stats.feature_dim]);
        let logits = model.forward(input_tensor);
        let logits_vec = logits.into_data().to_vec::<f32>().unwrap();

        let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut raw_probs = vec![0.0f32; logits_vec.len()];
        for (i, &l) in logits_vec.iter().enumerate() {
            let p = (l - max_l).exp();
            raw_probs[i] = p;
            exp_sum += p;
        }
        for p in &mut raw_probs {
            *p /= exp_sum;
        }

        let (stable_idx, _, _) = stabilizer.update(&raw_probs, total_ac_rms);
        confusion[label][stable_idx] += 1;
    }

    println!("\n=== 🎯 Stabilizer 포함 평가 오차 행렬 ===");
    for (i, row) in confusion.iter().enumerate() {
        let class_name = stats.class_names.get(i).cloned().unwrap_or_else(|| format!("Class {}", i));
        let correct = row[i];
        let total: usize = row.iter().sum();
        let acc = (correct as f32 / total as f32) * 100.0;
        println!("  정답 [{:<25}]: 정확도 {:5.1}% ({:4}/{}개) | 예측 분포: {:?}", class_name, acc, correct, total, row);
    }

    // 🔬 [심층 분석: 변동 및 스케일 변화 시 모델 예측 분포 테스트]
    println!("\n🔬 [심층 분석: 신호 변동 및 외삽(Extrapolation) 시 예측 반응]");
    let file = std::fs::File::open(dataset_path).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let mut rest_samples: Vec<Vec<f32>> = Vec::new();
    let mut fist_samples: Vec<Vec<f32>> = Vec::new();
    for res in rdr.records() {
        let rec = res.unwrap();
        let label: usize = rec[label_idx].parse().unwrap();
        let feats: Vec<f32> = (0..stats.feature_dim).map(|i| rec[i].parse().unwrap_or(0.0)).collect();
        if label == 0 {
            rest_samples.push(feats);
        } else if label == 2 {
            fist_samples.push(feats);
        }
    }

    let predict_dist = |feats: &[f32]| -> Vec<f32> {
        let norm = stats.normalize(feats);
        let input_tensor = Tensor::<MyBackend, 1>::from_floats(norm.as_slice(), &device)
            .reshape([1, stats.feature_dim]);
        let logits = model.forward(input_tensor);
        let logits_vec = logits.into_data().to_vec::<f32>().unwrap();
        let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut raw_probs = vec![0.0f32; logits_vec.len()];
        for (i, &l) in logits_vec.iter().enumerate() {
            let p = (l - max_l).exp();
            raw_probs[i] = p;
            exp_sum += p;
        }
        for p in &mut raw_probs {
            *p /= exp_sum;
        }
        raw_probs
    };

    // 1. 휴식기 평균 상태
    if !rest_samples.is_empty() {
        let mut mean_rest = vec![0.0f32; stats.feature_dim];
        for s in &rest_samples {
            for i in 0..stats.feature_dim {
                mean_rest[i] += s[i];
            }
        }
        for i in 0..stats.feature_dim {
            mean_rest[i] /= rest_samples.len() as f32;
        }

        let p_orig = predict_dist(&mean_rest);
        println!("  - 원본 휴식기 평균: [Rest: {:.1}%, Noise: {:.1}%, Fist: {:.1}%]", p_orig[0]*100.0, p_orig[1]*100.0, p_orig[2]*100.0);

        // 휴식기 신호가 +10% 커졌을 때 (미세 근육 긴장 / 착용 편차)
        let rest_plus10: Vec<f32> = mean_rest.iter().map(|&x| x * 1.10).collect();
        let p_plus10 = predict_dist(&rest_plus10);
        println!("  - 휴식기 +10% 증가: [Rest: {:.1}%, Noise: {:.1}%, Fist: {:.1}%]", p_plus10[0]*100.0, p_plus10[1]*100.0, p_plus10[2]*100.0);

        // 휴식기 신호가 +25% 커졌을 때
        let rest_plus25: Vec<f32> = mean_rest.iter().map(|&x| x * 1.25).collect();
        let p_plus25 = predict_dist(&rest_plus25);
        println!("  - 휴식기 +25% 증가: [Rest: {:.1}%, Noise: {:.1}%, Fist: {:.1}%]", p_plus25[0]*100.0, p_plus25[1]*100.0, p_plus25[2]*100.0);

        // 휴식기 신호가 -10% 작아졌을 때
        let rest_minus10: Vec<f32> = mean_rest.iter().map(|&x| x * 0.90).collect();
        let p_minus10 = predict_dist(&rest_minus10);
        println!("  - 휴식기 -10% 감소: [Rest: {:.1}%, Noise: {:.1}%, Fist: {:.1}%]", p_minus10[0]*100.0, p_minus10[1]*100.0, p_minus10[2]*100.0);
    }

    // 2. 진짜 주먹 평균 상태
    if !fist_samples.is_empty() {
        let mut mean_fist = vec![0.0f32; stats.feature_dim];
        for s in &fist_samples {
            for i in 0..stats.feature_dim {
                mean_fist[i] += s[i];
            }
        }
        for i in 0..stats.feature_dim {
            mean_fist[i] /= fist_samples.len() as f32;
        }

        let p_orig = predict_dist(&mean_fist);
        println!("  - 원본 주먹 평균  : [Rest: {:.1}%, Noise: {:.1}%, Fist: {:.1}%]", p_orig[0]*100.0, p_orig[1]*100.0, p_orig[2]*100.0);

        // 주먹을 살짝 약하게 쥐었을 때 (-15%)
        let fist_minus15: Vec<f32> = mean_fist.iter().map(|&x| x * 0.85).collect();
        let p_minus15 = predict_dist(&fist_minus15);
        println!("  - 주먹 약하게(-15%): [Rest: {:.1}%, Noise: {:.1}%, Fist: {:.1}%]", p_minus15[0]*100.0, p_minus15[1]*100.0, p_minus15[2]*100.0);

        // 주먹을 더 세게 쥐었을 때 (+20%)
        let fist_plus20: Vec<f32> = mean_fist.iter().map(|&x| x * 1.20).collect();
        let p_plus20 = predict_dist(&fist_plus20);
        println!("  - 주먹 세게(+20%)  : [Rest: {:.1}%, Noise: {:.1}%, Fist: {:.1}%]", p_plus20[0]*100.0, p_plus20[1]*100.0, p_plus20[2]*100.0);
    }
}
