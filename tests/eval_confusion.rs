use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::Tensor;
use emg_ml_pipeline::models::EmgMlpConfig;
use emg_ml_pipeline::train::{MyBackend, NormalizationStats};

#[test]
fn eval_mlp_confusion_matrix() {
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

    // CSV 원본 데이터 읽어서 정규화 후 예측
    let file = std::fs::File::open(dataset_path).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let headers = rdr.headers().unwrap().clone();
    let label_idx = headers.iter().position(|h| h == "label").unwrap();

    let mut total_count = 0;
    let mut confusion: Vec<Vec<usize>> = vec![vec![0; num_classes]; num_classes];

    for res in rdr.records() {
        let rec = res.unwrap();
        let label: usize = rec[label_idx].parse().unwrap();
        let feats: Vec<f32> = (0..stats.feature_dim).map(|i| rec[i].parse().unwrap_or(0.0)).collect();
        let norm_feats = stats.normalize(&feats);

        let input_tensor = Tensor::<MyBackend, 1>::from_floats(norm_feats.as_slice(), &device)
            .reshape([1, stats.feature_dim]);
        let logits = model.forward(input_tensor);
        let pred: i32 = logits.argmax(1).into_scalar();

        confusion[label][pred as usize] += 1;
        total_count += 1;
    }

    println!("\n=== 🎯 전체 데이터셋(2,170개) 모델 평가 오차 행렬 (Confusion Matrix) ===");
    for (i, row) in confusion.iter().enumerate() {
        let class_name = stats.class_names.get(i).cloned().unwrap_or_else(|| format!("Class {}", i));
        let correct = row[i];
        let total: usize = row.iter().sum();
        let acc = (correct as f32 / total as f32) * 100.0;
        println!("  정답 [{:<25}]: 정확도 {:5.1}% ({:4}/{}개) | 예측 분포: {:?}", class_name, acc, correct, total, row);
    }
    println!("총 샘플: {}개\n", total_count);
}

#[test]
fn eval_modular_mlp_realtime_simulation() {
    use emg_ml_pipeline::models::{EmgModularMlpModel, RealTimeRingBuffer, MODULAR_MLP_WINDOW_SIZE};

    let device = burn::backend::wgpu::WgpuDevice::default();
    let dataset_path = "emg_dataset_noise_robust_20261108_102522.csv";
    if !std::path::Path::new(dataset_path).exists() {
        return;
    }

    let stats = NormalizationStats::load_for_model("emg_modular_mlp_model")
        .expect("Failed to load norm stats");
    let num_classes = stats.num_classes;

    let mut model = EmgModularMlpModel::<MyBackend>::new(&device, stats.feature_dim, num_classes);
    let recorder = CompactRecorder::new();
    let rec = recorder
        .load("emg_modular_mlp_model".into(), &device)
        .expect("Failed to load weights");
    model = model.load_record(rec);

    let file = std::fs::File::open(dataset_path).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let headers = rdr.headers().unwrap().clone();
    let label_idx = headers.iter().position(|h| h == "label").unwrap();

    let mut ring_buffer = RealTimeRingBuffer::new(MODULAR_MLP_WINDOW_SIZE, stats.feature_dim, None, None);
    let mut confusion = vec![vec![0usize; num_classes]; num_classes];
    let mut total_count = 0;

    for res in rdr.records() {
        let rec = res.unwrap();
        let label: usize = rec[label_idx].parse().unwrap();
        let feats: Vec<f32> = (0..stats.feature_dim)
            .map(|i| rec[i].parse().unwrap_or(0.0))
            .collect();
        let norm_feats = stats.normalize(&feats);

        let input_3d = ring_buffer.push_normalized_and_get_tensor::<MyBackend>(&norm_feats, &device);
        let logits = model.forward(input_3d);
        let pred: i32 = logits.argmax(1).into_scalar();

        confusion[label][pred as usize] += 1;
        total_count += 1;
    }

    println!("\n=== 🎯 Block-MLP 실시간 링버퍼 순수 모델 평가 (총 {} 프레임) ===", total_count);
    for (i, row) in confusion.iter().enumerate() {
        let class_name = stats.class_names.get(i).cloned().unwrap_or_else(|| format!("Class {}", i));
        let correct = row[i];
        let total: usize = row.iter().sum();
        let acc = (correct as f32 / total as f32) * 100.0;
        println!("  정답 [{:<25}]: 정확도 {:5.1}% ({:4}/{}개) | 예측 분포: {:?}", class_name, acc, correct, total, row);
    }

    // 이제 main.rs의 GestureStabilizer를 통과시켰을 때 어떻게 변하는지 동일한 데이터로 검증!
    let file2 = std::fs::File::open(dataset_path).unwrap();
    let mut rdr2 = csv::Reader::from_reader(file2);
    let headers2 = rdr2.headers().unwrap().clone();
    let label_idx2 = headers2.iter().position(|h| h == "label").unwrap();
    let rms_indices2: Vec<usize> = headers2
        .iter()
        .enumerate()
        .filter(|(_, h)| h.ends_with("_rms"))
        .map(|(i, _)| i)
        .collect();

    let mut ring_buffer2 = RealTimeRingBuffer::new(MODULAR_MLP_WINDOW_SIZE, stats.feature_dim, None, None);
    // main.rs와 동일한 base_rest_rms (약 51.0)
    let mut stabilizer = tests_stabilizer::GestureStabilizer::new(num_classes, 51.0);
    let mut stab_confusion = vec![vec![0usize; num_classes]; num_classes];

    for res in rdr2.records() {
        let rec = res.unwrap();
        let label: usize = rec[label_idx2].parse().unwrap();
        let feats: Vec<f32> = (0..stats.feature_dim)
            .map(|i| rec[i].parse().unwrap_or(0.0))
            .collect();
        let total_ac_rms: f32 = rms_indices2.iter().map(|&idx| rec[idx].parse::<f32>().unwrap_or(0.0)).sum();
        let norm_feats = stats.normalize(&feats);

        let input_3d = ring_buffer2.push_normalized_and_get_tensor::<MyBackend>(&norm_feats, &device);
        let logits = model.forward(input_3d);
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
        stab_confusion[label][stable_idx] += 1;
    }

    println!("\n=== 🎯 Block-MLP + GestureStabilizer 실시간 시뮬레이션 평가 ===");
    for (i, row) in stab_confusion.iter().enumerate() {
        let class_name = stats.class_names.get(i).cloned().unwrap_or_else(|| format!("Class {}", i));
        let correct = row[i];
        let total: usize = row.iter().sum();
        let acc = (correct as f32 / total as f32) * 100.0;
        println!("  정답 [{:<25}]: 정확도 {:5.1}% ({:4}/{}개) | 예측 분포: {:?}", class_name, acc, correct, total, row);
    }

    // 🚀 주먹(Label 2) 구간 세부 프레임별 확률 분석
    println!("\n=== 🔍 [정밀 분석] 주먹(Label 2) 유지 구간 프레임별 확률 추이 ===");
    let file3 = std::fs::File::open(dataset_path).unwrap();
    let mut rdr3 = csv::Reader::from_reader(file3);
    let mut ring_buffer3 = RealTimeRingBuffer::new(MODULAR_MLP_WINDOW_SIZE, stats.feature_dim, None, None);
    let mut fist_frame_idx = 0;
    let mut current_block_idx = 0;
    let mut last_label = 999;

    for (row_idx, res) in rdr3.records().enumerate() {
        let rec = res.unwrap();
        let label: usize = rec[label_idx].parse().unwrap();
        let feats: Vec<f32> = (0..stats.feature_dim)
            .map(|i| rec[i].parse().unwrap_or(0.0))
            .collect();
        let norm_feats = stats.normalize(&feats);
        let input_3d = ring_buffer3.push_normalized_and_get_tensor::<MyBackend>(&norm_feats, &device);
        let logits = model.forward(input_3d);
        let logits_vec = logits.into_data().to_vec::<f32>().unwrap();

        let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut probs = vec![0.0f32; logits_vec.len()];
        for (i, &l) in logits_vec.iter().enumerate() {
            let p = (l - max_l).exp();
            probs[i] = p;
            exp_sum += p;
        }
        for p in &mut probs {
            *p /= exp_sum;
        }

        if label == 2 {
            if last_label != 2 {
                current_block_idx += 1;
                fist_frame_idx = 0;
                println!("\n--- 주먹 유지 블록 {} (Row {}) ---", current_block_idx, row_idx);
            }
            fist_frame_idx += 1;

            // 20프레임(~0.4초)마다, 또는 Noise(1)가 Fist(2)보다 높아지는 순간 출력
            let pred: usize = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if fist_frame_idx % 30 == 1 || pred == 1 {
                let status = if pred == 2 { "✔ FIST 유지" } else if pred == 1 { "⚠️ NOISE 역전!" } else { "❌ REST" };
                println!("  [Block {} | Frame {:3} (~{:4.1}초)]: Rest={:4.1}%, Noise={:4.1}%, Fist={:4.1}% -> {} (Pred={})",
                    current_block_idx, fist_frame_idx, (fist_frame_idx as f32) * 0.02,
                    probs[0]*100.0, probs[1]*100.0, probs[2]*100.0, status, pred
                );
            }
        }
        last_label = label;
    }
}

mod tests_stabilizer {
    use std::time::{Duration, Instant};

    pub struct GestureStabilizer {
        pub ema_probs: Vec<f32>,
        pub current_stable_gesture: usize,
        pub candidate_gesture: usize,
        pub candidate_count: usize,
        pub last_transition_time: Instant,
        pub refractory_period: Duration,
        pub fist_class_idx: usize,
    }

    impl GestureStabilizer {
        pub fn new(num_classes: usize, _base_rest_rms: f32) -> Self {
            let mut init_probs = vec![0.0f32; num_classes];
            if !init_probs.is_empty() {
                init_probs[0] = 1.0;
            }

            Self {
                ema_probs: init_probs,
                current_stable_gesture: 0,
                candidate_gesture: 0,
                candidate_count: 0,
                last_transition_time: Instant::now() - Duration::from_secs(1),
                refractory_period: Duration::from_millis(300),
                fist_class_idx: if num_classes == 3 { 2 } else { 1 },
            }
        }

        pub fn update(&mut self, raw_probs: &[f32], total_ac_rms: f32) -> (usize, f32, &'static str) {
            let num_classes = raw_probs.len();
            if self.ema_probs.len() != num_classes {
                self.ema_probs = vec![0.0; num_classes];
                self.ema_probs[0] = 1.0;
            }

            // 1. 순수 신경망 예측 확률 벡터 EMA 스무딩 (alpha = 0.25)
            for i in 0..num_classes {
                self.ema_probs[i] = 0.25 * raw_probs[i] + 0.75 * self.ema_probs[i];
            }

            // 2. 모터 불응기(Refractory Period) 검사
            if self.last_transition_time.elapsed() < self.refractory_period {
                let p = self.ema_probs.get(self.current_stable_gesture).copied().unwrap_or(0.0);
                return (self.current_stable_gesture, p, "잠금");
            }

            // 3. 최적 후보 탐색
            let mut best_cand = 0;
            let mut best_p = self.ema_probs[0];
            for (i, &p) in self.ema_probs.iter().enumerate().skip(1) {
                if p > best_p {
                    best_p = p;
                    best_cand = i;
                }
            }

            // 4. 비대칭 히스테리시스 및 생체역학적 에너지 인터락
            let mut state_tag;
            if best_cand == self.current_stable_gesture {
                self.candidate_gesture = best_cand;
                self.candidate_count = 0;
                state_tag = "안정";
            } else {
                let is_current_fist = self.current_stable_gesture == self.fist_class_idx;
                let is_cand_fist = best_cand == self.fist_class_idx;
                let fist_p = self.ema_probs.get(self.fist_class_idx).copied().unwrap_or(0.0);
                let required_debounce = if is_current_fist { 5 } else { 3 };

                let valid_transition = if is_cand_fist {
                    best_p >= 0.60 && total_ac_rms >= 208.0 * 1.16
                } else if is_current_fist {
                    // 수축 에너지 유지 중(RMS >= 208 * 1.18 = 245)에는 동적 잡음(1)으로의 탈출을 차단
                    if total_ac_rms >= 208.0 * 1.18 {
                        if best_cand != 0 {
                            false
                        } else {
                            best_p >= 0.60 && fist_p < 0.28
                        }
                    } else if total_ac_rms < 208.0 * 1.12 {
                        fist_p < 0.45
                    } else {
                        fist_p < 0.28
                    }
                } else {
                    best_p >= 0.50
                };

                if valid_transition {
                    let same_direction = if is_current_fist && !is_cand_fist {
                        self.candidate_gesture != self.fist_class_idx
                    } else {
                        best_cand == self.candidate_gesture
                    };

                    if same_direction && self.candidate_count > 0 {
                        self.candidate_gesture = best_cand;
                        self.candidate_count += 1;
                    } else {
                        self.candidate_gesture = best_cand;
                        self.candidate_count = 1;
                    }
                    state_tag = "전이";

                    if self.candidate_count >= required_debounce {
                        self.current_stable_gesture = best_cand;
                        self.candidate_count = 0;
                        self.last_transition_time = Instant::now();
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
}

