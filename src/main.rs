use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::Tensor;
use burn_wgpu::WgpuDevice;
use emg_ml_pipeline::{
    fft::{EmgFftProcessor, NUM_CHANNELS, TOTAL_FEATURES_ALL, TOTAL_FEATURES_WITH_RAW},
    models::{EmgCnnLstmModel, EmgCnnModel, EmgLstmModel, EmgTcnModel, ModelArchitecture},
    train::{
        train_model, MyBackend, NormalizationStats, TrainingConfig, TrainingProgress, NUM_CLASSES,
        SEQ_LEN,
    },
};
use glob::glob;
use reqwest::blocking::{multipart, Client};
use serde_json::Value;
use serialport;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::{Duration, Instant};

struct ActionStep {
    name: &'static str,
    label_id: usize,
    duration_secs: u64,
}

fn main() {
    let device = WgpuDevice::default();

    loop {
        println!("\n========================================================");
        println!(" 🦾 EMG 고성능 딥러닝 & 지표 추출 파이프라인 (SEQ_LEN: {})", SEQ_LEN);
        println!("========================================================");
        println!(" 1. 📥 데이터 수집 (원시값 + FFT + Hudgins 4대 지표 동시 저장)");
        println!(" 2. 🧠 로컬 모델 학습 (1D-TCN / 1D-CNN / LSTM / CNN+LSTM)");
        println!(" 3. 🚀 실시간 추론 (선택 모델 로드 & 실시간 복합 추론)");
        println!(" 4. 🌐 서버 원격 학습 및 모델 다운로드 (순수 Rust 서버 연동)");
        println!(" 5. 🚪 프로그램 종료");
        print!("메뉴를 선택하세요 (1-5) > ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            continue;
        }

        match input.trim() {
            "1" => {
                if let Err(e) = run_data_collection() {
                    eprintln!("❌ 데이터 수집 에러: {}", e);
                }
            }
            "2" => {
                if let Err(e) = run_local_training(&device) {
                    eprintln!("❌ 로컬 학습 에러: {}", e);
                }
            }
            "3" => {
                run_inference_pipeline(&device);
            }
            "4" => {
                if let Err(e) = run_remote_server_pipeline() {
                    eprintln!("❌ 서버 연동 에러: {}", e);
                }
            }
            "5" => {
                println!("프로그램을 종료합니다.");
                break;
            }
            _ => {
                println!("❌ 올바른 번호를 입력하세요 (1~5)");
            }
        }
    }
}

// =========================================================================
// 1. 센서 원시값 + FFT 스펙트럼 + 확장 시간 도메인 6대 지표(총 75차원) 동시 수집
// =========================================================================
fn run_data_collection() -> io::Result<()> {
    let port_name = "/dev/ttyACM0";
    let baud_rate = 9600;

    println!("\n시리얼 포트 연결 중: {}...", port_name);
    let port = serialport::new(port_name, baud_rate)
        .timeout(Duration::from_millis(1000))
        .open()
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e))?;

    let mut reader = BufReader::new(port);
    let file_name = format!(
        "emg_dataset_{}.csv",
        chrono::Local::now().format("%Y%m%d_%H%M%S")
    );
    let mut csv_file = File::create(&file_name)?;

    // 75차원 전체 특징 CSV 헤더 작성 (원시 5 + FFT 40 + 시간 도메인 30 = 75)
    writeln!(csv_file, "{}", EmgFftProcessor::csv_header())?;
    println!("저장 파일 생성: {} (특징 총 75차원: 원시 5 + FFT 40 + 시간 도메인 30)\n", file_name);

    let steps = vec![
        ActionStep { name: "휴식 (Relax)", label_id: 0, duration_secs: 7 },
        ActionStep { name: "주먹 쥐기 (Fist)", label_id: 1, duration_secs: 7 },
        ActionStep { name: "손가락 펴기 (Open)", label_id: 2, duration_secs: 7 },
        ActionStep { name: "엄지 굽히기 (Thumb Flex)", label_id: 3, duration_secs: 7 },
        ActionStep { name: "휴식 (Relax)", label_id: 0, duration_secs: 7 },
    ];

    println!("=== 💡 중요 가이드 ===");
    println!("카운트다운이 시작되기 **전**에 미리 해당 자세를 취하고 계세요!");
    println!("시작 직후 2초 동안은 움직임 노이즈 방지를 위해 자동 제외됩니다.");
    println!("준비되셨으면 엔터 키를 누르세요...");
    let mut dummy = String::new();
    io::stdin().read_line(&mut dummy)?;

    let mut processor = EmgFftProcessor::new();

    for (idx, step) in steps.iter().enumerate() {
        println!("\n--------------------------------------------------");
        println!("[ 단계 {} / {} ]", idx + 1, steps.len());
        println!("👉 미리 자세 잡기: ** {} ** (잠시 후 시작)", step.name);
        println!("--------------------------------------------------");

        for i in (1..=3).rev() {
            print!("{}... ", i);
            io::stdout().flush()?;
            std::thread::sleep(Duration::from_secs(1));
        }
        println!("측정 시작! (자세 고정 🚀)");

        let start_time = Instant::now();
        let transition_buffer = Duration::from_secs(2);
        let total_duration = Duration::from_secs(step.duration_secs);
        let mut is_recording_started = false;

        while start_time.elapsed() < total_duration {
            let mut line = String::new();
            if let Ok(_) = reader.read_line(&mut line) {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if start_time.elapsed() < transition_buffer {
                    print!("\r⏳ [과도기 대기] 자세 유지 중... [{}]     ", trimmed);
                    io::stdout().flush()?;
                    continue;
                }

                if !is_recording_started {
                    println!("\n✔ [순수 데이터 수집 중] 원시값 + FFT + 확장 시간 도메인 75차원 지표 기록 시작!");
                    is_recording_started = true;
                }

                let parts: Vec<f32> = trimmed
                    .split(',')
                    .filter_map(|s| s.trim().parse::<f32>().ok())
                    .collect();

                if parts.len() == NUM_CHANNELS {
                    let mut raw_arr = [0.0f32; NUM_CHANNELS];
                    raw_arr.copy_from_slice(&parts[..NUM_CHANNELS]);

                    // 실시간 원시(5) + FFT(40) + 시간 도메인(30) = 75차원 통합 산출
                    let all_features = processor.process_sample_all(&raw_arr);

                    let mut row_str = String::new();
                    for (i, val) in all_features.iter().enumerate() {
                        if i > 0 {
                            row_str.push(',');
                        }
                        row_str.push_str(&format!("{:.2}", val));
                    }
                    row_str.push_str(&format!(",{},{}\n", step.label_id, step.name));

                    csv_file.write_all(row_str.as_bytes())?;
                    print!(
                        "\r[기록 중] 원시:[{:.0},{:.0},{:.0},{:.0},{:.0}] | 75차원 지표 연산 완료     ",
                        raw_arr[0], raw_arr[1], raw_arr[2], raw_arr[3], raw_arr[4]
                    );
                    io::stdout().flush()?;
                }
            }
        }
        println!("\n✔ 해당 동작 수집 완료!");
    }

    println!("\n🎉 고품질 다차원(75차원) 데이터 수집 완료! 파일: {}", file_name);
    Ok(())
}

// =========================================================================
// 2. 로컬 모델 학습 (1D-TCN / 1D-CNN / LSTM / CNN+LSTM)
// =========================================================================
fn run_local_training(device: &WgpuDevice) -> io::Result<()> {
    let mut found_files = Vec::new();
    for entry in glob("emg_dataset_*.csv").expect("Failed glob") {
        if let Ok(path) = entry {
            found_files.push(path);
        }
    }
    for entry in glob("uploads/*.csv").expect("Failed glob") {
        if let Ok(path) = entry {
            found_files.push(path);
        }
    }

    if found_files.is_empty() {
        println!("\n❌ [에러] 수집된 데이터셋 파일이 없습니다. 1번 데이터 수집을 먼저 실행해주세요.");
        return Ok(());
    }

    println!("\n[데이터셋 목록]");
    for (i, file) in found_files.iter().enumerate() {
        println!("  [{}] {}", i + 1, file.display());
    }
    print!("학습에 사용할 파일 번호 (기본 1번) > ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice = input.trim().parse::<usize>().unwrap_or(1);
    let dataset_path = if choice > 0 && choice <= found_files.len() {
        &found_files[choice - 1]
    } else {
        &found_files[0]
    };

    println!("\n[학습할 모델 아키텍처 선택]");
    println!("  [1] 1D-TCN (🥇 추천! Dilated Residual Network)");
    println!("  [2] 1D-CNN (BatchNorm 적용)");
    println!("  [3] LSTM");
    println!("  [4] CNN + LSTM 복합");
    println!("  [5] 4종 모델 모두 순차 학습");
    print!("선택 (1-5, 기본 1) > ");
    io::stdout().flush()?;

    let mut model_input = String::new();
    io::stdin().read_line(&mut model_input)?;
    let model_choice = model_input.trim();

    let models_to_train = match model_choice {
        "2" => vec![ModelArchitecture::Cnn],
        "3" => vec![ModelArchitecture::Lstm],
        "4" => vec![ModelArchitecture::CnnLstm],
        "5" => vec![
            ModelArchitecture::Tcn,
            ModelArchitecture::Cnn,
            ModelArchitecture::Lstm,
            ModelArchitecture::CnnLstm,
        ],
        _ => vec![ModelArchitecture::Tcn],
    };

    for arch in models_to_train {
        println!("\n========================================================");
        println!(" 🚀 {} 모델 GPU 학습 시작 (데이터셋: {})", arch.display_name(), dataset_path.display());
        println!("========================================================");

        let config = TrainingConfig {
            dataset_path: dataset_path.to_string_lossy().to_string(),
            model_arch: arch,
            epochs: 150,
            learning_rate: 1e-3, // 안정화된 0.001 학습률
            batch_size: 32,
        };

        let start_time = Instant::now();
        let saved_file = train_model(device, &config, |prog: TrainingProgress| {
            if prog.epoch % 15 == 0 || prog.epoch == 1 {
                println!(
                    "  ➔ [Epoch {:3}/{:3}] Loss: {:.4}",
                    prog.epoch, prog.total_epochs, prog.loss
                );
            }
        })?;

        println!(
            "💾 {} 학습 완료! (소요: {:.1}초) 모델 저장: {}",
            arch.display_name(),
            start_time.elapsed().as_secs_f32(),
            saved_file
        );
    }

    Ok(())
}

// =========================================================================
// 3. 실시간 추론 (모델 선택 및 실시간 복합 추론)
// =========================================================================
fn run_inference_pipeline(device: &WgpuDevice) {
    println!("\n[추론에 사용할 모델 선택]");
    println!("  [1] 1D-TCN (emg_tcn_model.mpk) [추천]");
    println!("  [2] 1D-CNN (emg_cnn_model.mpk)");
    println!("  [3] LSTM (emg_lstm_model.mpk)");
    println!("  [4] CNN + LSTM (emg_cnnlstm_model.mpk)");
    print!("선택 (1-4, 기본 1) > ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let arch = match input.trim() {
        "2" => ModelArchitecture::Cnn,
        "3" => ModelArchitecture::Lstm,
        "4" => ModelArchitecture::CnnLstm,
        _ => ModelArchitecture::Tcn,
    };

    let model_file_name = arch.file_name();
    let model_file_path = format!("{}.mpk", model_file_name);
    if !Path::new(&model_file_path).exists() {
        println!(
            "\n❌ [에러] '{}' 모델 파일이 없습니다. 먼저 학습을 진행해주세요.",
            model_file_path
        );
        return;
    }

    // 특징 차원 선택 (75차원 전체, 45차원, 5차원)
    println!("\n[입력 특징 차원 설정]");
    println!("  [1] 75차원 (원시 5 + FFT 40 + 시간 도메인 30) [최신 신규]");
    println!("  [2] 45차원 (원시 5 + FFT 40)");
    println!("  [3] 5차원 (원시 신호만)");
    print!("선택 (1-3, 기본 1) > ");
    io::stdout().flush().unwrap();
    let mut feat_input = String::new();
    io::stdin().read_line(&mut feat_input).unwrap();
    let feature_dim = match feat_input.trim() {
        "2" => TOTAL_FEATURES_WITH_RAW,
        "3" => NUM_CHANNELS,
        _ => TOTAL_FEATURES_ALL,
    };

    println!("\n[알림] 저장된 {} 모델 로드 중 (특징 차원: {})...", arch.display_name(), feature_dim);
    let recorder = CompactRecorder::new();

    enum LoadedModel {
        Tcn(EmgTcnModel<MyBackend>),
        Cnn(EmgCnnModel<MyBackend>),
        Lstm(EmgLstmModel<MyBackend>),
        CnnLstm(EmgCnnLstmModel<MyBackend>),
    }

    let loaded_model = match arch {
        ModelArchitecture::Tcn => {
            let mut m = EmgTcnModel::new(device, feature_dim, NUM_CLASSES);
            match recorder.load(model_file_name.into(), device) {
                Ok(rec) => m = m.load_record(rec),
                Err(e) => {
                    eprintln!("❌ 모델 로드 실패: {:?}", e);
                    return;
                }
            }
            LoadedModel::Tcn(m)
        }
        ModelArchitecture::Cnn => {
            let mut m = EmgCnnModel::new(device, feature_dim, NUM_CLASSES);
            match recorder.load(model_file_name.into(), device) {
                Ok(rec) => m = m.load_record(rec),
                Err(e) => {
                    eprintln!("❌ 모델 로드 실패: {:?}", e);
                    return;
                }
            }
            LoadedModel::Cnn(m)
        }
        ModelArchitecture::Lstm => {
            let mut m = EmgLstmModel::new(device, feature_dim, NUM_CLASSES);
            match recorder.load(model_file_name.into(), device) {
                Ok(rec) => m = m.load_record(rec),
                Err(e) => {
                    eprintln!("❌ 모델 로드 실패: {:?}", e);
                    return;
                }
            }
            LoadedModel::Lstm(m)
        }
        ModelArchitecture::CnnLstm => {
            let mut m = EmgCnnLstmModel::new(device, feature_dim, NUM_CLASSES);
            match recorder.load(model_file_name.into(), device) {
                Ok(rec) => m = m.load_record(rec),
                Err(e) => {
                    eprintln!("❌ 모델 로드 실패: {:?}", e);
                    return;
                }
            }
            LoadedModel::CnnLstm(m)
        }
    };

    println!("✔ {} 모델 로드 완료!", arch.display_name());

    let port_name = "/dev/ttyACM0";
    let baud_rate = 9600;

    println!("시리얼 포트 연결 중: {}...", port_name);
    let port = match serialport::new(port_name, baud_rate)
        .timeout(Duration::from_millis(500))
        .open()
    {
        Ok(p) => p,
        Err(e) => {
            eprintln!("❌ 시리얼 포트 연결 실패: {}", e);
            return;
        }
    };

    // 학습 시 저장된 Z-Score 정규화 통계 로드
    let norm_stats = NormalizationStats::load_for_model(model_file_name);
    if norm_stats.is_some() {
        println!("✔ [정규화 통계 로드 완료] 학습 당시의 정밀 Z-Score 표준화 적용 (오차 왜곡 제거)");
    } else {
        println!("ℹ️ 정규화 통계 파일 없음 (기본 스케일링 모드로 작동)");
    }

    let mut reader = BufReader::new(port);
    let action_names = [
        "휴식 (Relax)",
        "주먹 쥐기 (Fist)",
        "손가락 펴기 (Open)",
        "엄지 굽히기 (Thumb)",
    ];

    // 🚀 [1.5초 휴식기 영점 자동 보정]
    println!("\n⚖️ [센서 영점 보정] 팔에 힘을 빼고 편안히 1.5초간 가만히 유지하세요...");
    let mut baseline_samples: Vec<[f32; NUM_CHANNELS]> = Vec::new();
    let cal_start = std::time::Instant::now();
    while baseline_samples.len() < 75 && cal_start.elapsed().as_secs_f32() < 4.0 {
        let mut cal_line = String::new();
        if reader.read_line(&mut cal_line).is_ok() {
            let parts: Vec<f32> = cal_line
                .trim()
                .split(',')
                .filter_map(|s| s.trim().parse::<f32>().ok())
                .collect();
            if parts.len() == NUM_CHANNELS {
                let mut arr = [0.0f32; NUM_CHANNELS];
                arr.copy_from_slice(&parts);
                baseline_samples.push(arr);
                if baseline_samples.len() % 15 == 0 {
                    print!("\r  영점 측정 중... [{:2}/75 샘플]", baseline_samples.len());
                    io::stdout().flush().unwrap();
                }
            }
        }
    }
    let mut rest_baseline = [0.0f32; NUM_CHANNELS];
    if !baseline_samples.is_empty() {
        for s in &baseline_samples {
            for ch in 0..NUM_CHANNELS {
                rest_baseline[ch] += s[ch];
            }
        }
        for ch in 0..NUM_CHANNELS {
            rest_baseline[ch] /= baseline_samples.len() as f32;
        }
        println!(
            "\n✔ [영점 보정 완료] 현재 착용 기준선: [Ch0: {:.0}, Ch1: {:.0}, Ch2: {:.0}, Ch3: {:.0}, Ch4: {:.0}]",
            rest_baseline[0], rest_baseline[1], rest_baseline[2], rest_baseline[3], rest_baseline[4]
        );
    } else {
        println!("\nℹ️ [영점 보정 건너뜀] 즉시 추론을 시작합니다.");
    }

/// 🚀 실시간 채터링 및 플리커링을 원천 제거하는 지능형 제스처 안정화 엔진
/// 1. 확률 지수이동평균(EMA): 확률 벡터 자체를 부드럽게 LPF 필터링
/// 2. 히스테리시스 락: 신규 제스처 진입 임계치(0.70)와 현재 제스처 유지 임계치(0.40) 분리
/// 3. 상태 디바운싱: 연속 5프레임(~100ms) 이상 유지 시에만 상태 전이
pub struct GestureStabilizer {
    ema_probs: Vec<f32>,
    current_stable_gesture: usize,
    candidate_gesture: usize,
    candidate_count: usize,
    alpha: f32,            // EMA 갱신 계수 (0.25)
    enter_threshold: f32,  // 신규 진입 문턱 (70%)
    hold_threshold: f32,   // 기존 유지 문턱 (40%)
    debounce_frames: usize,// 디바운스 프레임 수 (5 frames @ 50Hz = 100ms)
}

impl GestureStabilizer {
    pub fn new(num_classes: usize) -> Self {
        let mut init_probs = vec![0.0f32; num_classes];
        if !init_probs.is_empty() {
            init_probs[0] = 1.0; // 초기 상태는 Rest
        }
        Self {
            ema_probs: init_probs,
            current_stable_gesture: 0,
            candidate_gesture: 0,
            candidate_count: 0,
            alpha: 0.25,
            enter_threshold: 0.70,
            hold_threshold: 0.40,
            debounce_frames: 5,
        }
    }

    pub fn update(&mut self, raw_probs: &[f32], is_at_rest: bool) -> (usize, f32, &'static str) {
        let num_classes = raw_probs.len();
        if self.ema_probs.len() != num_classes {
            self.ema_probs = vec![0.0; num_classes];
            self.ema_probs[0] = 1.0;
        }

        // 1. 순수 신경망 예측 확률 벡터 지수이동평균(EMA) 필터링 (인위적 확률 조작 제거)
        for i in 0..num_classes {
            self.ema_probs[i] = self.alpha * raw_probs[i] + (1.0 - self.alpha) * self.ema_probs[i];
        }

        // 2. 스마트 에너지 게이트 (근육 에너지가 극단적 데드존 미만일 때만 안전하게 휴식으로 분류)
        if is_at_rest {
            self.current_stable_gesture = 0;
            self.candidate_gesture = 0;
            self.candidate_count = 0;
            let rest_p = self.ema_probs.get(0).copied().unwrap_or(0.0);
            return (0, rest_p, "휴식");
        }

        // 3. 가장 높은 스무딩 확률을 가진 후보 제스처 탐색
        let mut best_cand = 0;
        let mut best_p = self.ema_probs[0];
        for (i, &p) in self.ema_probs.iter().enumerate().skip(1) {
            if p > best_p {
                best_p = p;
                best_cand = i;
            }
        }

        // 4. 히스테리시스 및 디바운싱 판단
        let mut state_tag;
        if best_cand == self.current_stable_gesture {
            // 현재 제스처가 여전히 우세함 -> 후보 카운터 리셋
            self.candidate_gesture = best_cand;
            self.candidate_count = 0;
            state_tag = "안정";
        } else {
            // 다른 제스처로 전환 시도
            let current_hold_p = self.ema_probs.get(self.current_stable_gesture).copied().unwrap_or(0.0);
            
            // 전이 조건: 새 제스처의 확률이 진입 임계치(70%)를 넘고, 현재 제스처의 확률이 유지 문턱(40%) 아래로 떨어졌을 때
            if best_p >= self.enter_threshold && current_hold_p < self.hold_threshold {
                if best_cand == self.candidate_gesture {
                    self.candidate_count += 1;
                } else {
                    self.candidate_gesture = best_cand;
                    self.candidate_count = 1;
                }
                state_tag = "전이";

                // 디바운스 프레임(5프레임 = 100ms) 이상 일관되게 유지되면 최종 상태 전이 확정
                if self.candidate_count >= self.debounce_frames {
                    self.current_stable_gesture = best_cand;
                    self.candidate_count = 0;
                    state_tag = "확정";
                }
            } else {
                // 불확실하거나 애매한 과도기 구간 -> 이전 안정 제스처를 그대로 락(Lock)
                self.candidate_count = 0;
                state_tag = "유지";
            }
        }

        let stable_p = self.ema_probs.get(self.current_stable_gesture).copied().unwrap_or(0.0);
        (self.current_stable_gesture, stable_p, state_tag)
    }

    /// 각 클래스별 스무딩된 실시간 확률 벡터 반환
    pub fn get_probabilities(&self) -> &[f32] {
        &self.ema_probs
    }
}

    let mut processor = EmgFftProcessor::new();
    let mut window_buffer: VecDeque<Vec<f32>> = VecDeque::with_capacity(SEQ_LEN);
    // 🚀 [지능형 제스처 안정화 엔진] 확률 EMA + 히스테리시스 락 + 100ms 디바운스 필터
    let mut stabilizer = GestureStabilizer::new(NUM_CLASSES);

    println!("\n==========================================================");
    println!(" 🚀 {} 실시간 추론 시작! (확률 EMA & 히스테리시스 락 적용 / 종료: Ctrl + C)", arch.display_name());
    println!("==========================================================");

    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let parts: Vec<f32> = trimmed
                    .split(',')
                    .filter_map(|s| s.trim().parse::<f32>().ok())
                    .collect();

                if parts.len() == NUM_CHANNELS {
                    let mut raw_arr = [0.0f32; NUM_CHANNELS];
                    raw_arr.copy_from_slice(&parts[..NUM_CHANNELS]);

                    // 설정된 차원에 맞게 특징 벡터 계산 (75차원 DC-Invariant AC 특징)
                    let current_features: Vec<f32> = match feature_dim {
                        TOTAL_FEATURES_ALL => processor.process_sample_all(&raw_arr).to_vec(),
                        TOTAL_FEATURES_WITH_RAW => {
                            processor.process_sample_combined(&raw_arr).to_vec()
                        }
                        _ => raw_arr.to_vec(),
                    };

                    // 🚀 [스마트 에너지 노이즈 게이트]
                    // 팔에 힘을 뺀 상태(휴식)에서는 5채널 총 AC-RMS 에너지가 75 미만임
                    let total_ac_rms = if feature_dim == TOTAL_FEATURES_ALL {
                        let feats_arr: &[f32; TOTAL_FEATURES_ALL] = current_features
                            .as_slice()
                            .try_into()
                            .unwrap();
                        EmgFftProcessor::extract_total_ac_rms(feats_arr)
                    } else {
                        100.0
                    };
                    let is_at_rest = total_ac_rms < 35.0;

                    // 🚀 [학습 시와 100% 동일한 정규화 적용]
                    let normalized: Vec<f32> = if let Some(ref stats) = norm_stats {
                        stats.normalize(&current_features)
                    } else {
                        // fallback: 원시값 512 기준, FFT 0 기준
                        current_features
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| if i < 5 { (v - 512.0) / 250.0 } else { v / 50.0 })
                            .collect()
                    };

                    if window_buffer.len() == SEQ_LEN {
                        window_buffer.pop_front();
                    }
                    window_buffer.push_back(normalized);

                    let mut seq_data = Vec::with_capacity(SEQ_LEN * feature_dim);
                    let first_elem = window_buffer.front().unwrap().clone();
                    for _ in 0..(SEQ_LEN - window_buffer.len()) {
                        seq_data.extend_from_slice(&first_elem);
                    }
                    for sample in &window_buffer {
                        seq_data.extend_from_slice(sample);
                    }

                    let input_tensor: Tensor<MyBackend, 3> = Tensor::<MyBackend, 1>::from_floats(
                        seq_data.as_slice(),
                        device,
                    )
                    .reshape([1, SEQ_LEN, feature_dim]);

                    let logits = match &loaded_model {
                        LoadedModel::Tcn(m) => m.forward(input_tensor),
                        LoadedModel::Cnn(m) => m.forward(input_tensor),
                        LoadedModel::Lstm(m) => m.forward(input_tensor),
                        LoadedModel::CnnLstm(m) => m.forward(input_tensor),
                    };

                    let logits_vec = logits.into_data().to_vec::<f32>().unwrap();

                    // 🚀 Softmax 확률 계산
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

                    // 🚀 지능형 제스처 안정화 (확률 EMA + 히스테리시스 락 + 100ms 디바운스)
                    let (stable_idx, _stable_p, state_tag) = stabilizer.update(&raw_probs, is_at_rest);
                    let probs = stabilizer.get_probabilities();
                    let p_relax = probs.get(0).copied().unwrap_or(0.0) * 100.0;
                    let p_fist  = probs.get(1).copied().unwrap_or(0.0) * 100.0;
                    let p_open  = probs.get(2).copied().unwrap_or(0.0) * 100.0;
                    let p_thumb = probs.get(3).copied().unwrap_or(0.0) * 100.0;

                    print!(
                        "\r🤖 [{:<6}] [{:>2}] {:<16} | [휴식:{:4.1}% | 주먹:{:4.1}% | 펴기:{:4.1}% | 엄지:{:4.1}%] | RMS:{:4.1}  ",
                        arch.display_name(),
                        state_tag,
                        action_names.get(stable_idx).unwrap_or(&"알 수 없음"),
                        p_relax,
                        p_fist,
                        p_open,
                        p_thumb,
                        total_ac_rms,
                    );
                    io::stdout().flush().unwrap();
                }
            }
            Err(_) => {
                continue;
            }
        }
    }
}

// =========================================================================
// 4. 순수 Rust 서버 연동 (데이터셋 업로드 -> 원격 학습 -> 모델 다운로드)
// =========================================================================
fn run_remote_server_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    print!("\n서버 주소를 입력하세요 (기본: http://127.0.0.1:3000) > ");
    io::stdout().flush()?;
    let mut server_url = String::new();
    io::stdin().read_line(&mut server_url)?;
    let mut server_url = server_url.trim().to_string();
    if server_url.is_empty() {
        server_url = "http://127.0.0.1:3000".to_string();
    }

    let client = Client::builder().timeout(Duration::from_secs(30)).build()?;

    // 1. 헬스 체크
    print!("서버 연결 확인 중 ({})... ", server_url);
    io::stdout().flush()?;
    match client.get(format!("{}/health", server_url)).send() {
        Ok(res) if res.status().is_success() => {
            println!("✔ 성공 (서버 가동 중)");
        }
        Ok(res) => {
            return Err(format!("서버 응답 비정상 (상태 코드: {})", res.status()).into());
        }
        Err(e) => {
            return Err(format!("서버에 연결할 수 없습니다: {}", e).into());
        }
    }

    // 2. 업로드할 로컬 데이터셋 선택
    let mut found_files = Vec::new();
    for entry in glob("emg_dataset_*.csv").expect("Failed glob") {
        if let Ok(path) = entry {
            found_files.push(path);
        }
    }

    if found_files.is_empty() {
        return Err("업로드할 로컬 데이터셋(emg_dataset_*.csv)이 없습니다.".into());
    }

    println!("\n[업로드할 데이터셋 선택]");
    for (i, f) in found_files.iter().enumerate() {
        println!("  [{}] {}", i + 1, f.display());
    }
    print!("선택 번호 (기본 1) > ");
    io::stdout().flush()?;

    let mut choice_input = String::new();
    io::stdin().read_line(&mut choice_input)?;
    let choice = choice_input.trim().parse::<usize>().unwrap_or(1);
    let selected_file = if choice > 0 && choice <= found_files.len() {
        &found_files[choice - 1]
    } else {
        &found_files[0]
    };

    // 3. 서버로 데이터셋 업로드
    println!("\n📤 데이터셋 업로드 중: {}...", selected_file.display());
    let form = multipart::Form::new().file("file", selected_file)?;
    let upload_res = client
        .post(format!("{}/api/upload", server_url))
        .multipart(form)
        .send()?;

    if !upload_res.status().is_success() {
        let err_text = upload_res.text()?;
        return Err(format!("업로드 실패: {}", err_text).into());
    }

    let upload_json: Value = upload_res.json()?;
    let uploaded_name = upload_json["filename"]
        .as_str()
        .unwrap_or(selected_file.file_name().unwrap().to_str().unwrap());
    println!("✔ 업로드 완료! 서버 저장 파일명: {}", uploaded_name);

    // 4. 서버에서 학습할 모델 선택
    println!("\n[서버에서 학습할 모델 아키텍처 선택]");
    println!("  [1] 1D-TCN (🥇 추천! Temporal Convolutional Network) (`tcn`)");
    println!("  [2] 1D-CNN (`cnn`)");
    println!("  [3] LSTM (`lstm`)");
    println!("  [4] CNN + LSTM (`cnn_lstm`)");
    print!("선택 (1-4, 기본 1) > ");
    io::stdout().flush()?;

    let mut model_choice = String::new();
    io::stdin().read_line(&mut model_choice)?;
    let model_type_str = match model_choice.trim() {
        "2" => "cnn",
        "3" => "lstm",
        "4" => "cnn_lstm",
        _ => "tcn",
    };

    // 5. 서버 학습 시작 요청 (기본 lr 0.001)
    println!("\n🧠 서버에 원격 학습 시작 요청 중 (Learning Rate: 0.001)...");
    let train_req = serde_json::json!({
        "dataset": uploaded_name,
        "model_type": model_type_str,
        "epochs": 150,
        "lr": 0.001,
        "batch_size": 32
    });

    let train_res = client
        .post(format!("{}/api/train", server_url))
        .json(&train_req)
        .send()?;

    if !train_res.status().is_success() {
        let err_text = train_res.text()?;
        return Err(format!("학습 시작 요청 실패: {}", err_text).into());
    }
    println!("✔ 서버 학습이 정상적으로 시작되었습니다!");

    // 6. 실시간 학습 상태 모니터링
    println!("\n--- 서버 원격 학습 모니터링 진행 중 ---");
    let mut last_epoch = 0;
    let completed_model_file;

    loop {
        std::thread::sleep(Duration::from_millis(800));
        let status_res = match client.get(format!("{}/api/status", server_url)).send() {
            Ok(r) => r,
            Err(_) => continue,
        };

        if let Ok(status_json) = status_res.json::<Value>() {
            let status = status_json["status"].as_str().unwrap_or("unknown");
            let epoch = status_json["epoch"].as_u64().unwrap_or(0) as usize;
            let total_epochs = status_json["total_epochs"].as_u64().unwrap_or(150) as usize;
            let loss = status_json["loss"].as_f64().unwrap_or(0.0) as f32;

            if epoch != last_epoch && epoch > 0 {
                println!(
                    "  ➔ [서버 진행률] Epoch [{:3}/{:3}] | 평균 Loss: {:.4}",
                    epoch, total_epochs, loss
                );
                last_epoch = epoch;
            }

            if status == "completed" {
                completed_model_file = status_json["model_file"]
                    .as_str()
                    .map(|s| s.to_string());
                println!("\n🎉 서버에서 모델 학습이 성공적으로 완료되었습니다!");
                break;
            } else if status == "failed" {
                let err_msg = status_json["error"].as_str().unwrap_or("알 수 없는 오류");
                return Err(format!("서버 학습 실패: {}", err_msg).into());
            }
        }
    }

    // 7. 학습된 모델 다운로드
    let target_model_name = completed_model_file.unwrap_or_else(|| {
        match model_type_str {
            "cnn" => "emg_cnn_model.mpk".to_string(),
            "cnn_lstm" => "emg_cnnlstm_model.mpk".to_string(),
            "lstm" => "emg_lstm_model.mpk".to_string(),
            _ => "emg_tcn_model.mpk".to_string(),
        }
    });

    println!("\n📥 학습된 모델 다운로드 중: {}...", target_model_name);
    let download_url = format!("{}/api/download/{}", server_url, target_model_name);
    let mut dl_res = client.get(&download_url).send()?;

    if !dl_res.status().is_success() {
        return Err(format!("모델 다운로드 실패: {}", dl_res.status()).into());
    }

    let mut dest_file = File::create(&target_model_name)?;
    io::copy(&mut dl_res, &mut dest_file)?;
    println!(
        "💾 모델 다운로드 성공! 로컬에 저장되었습니다: {}",
        target_model_name
    );

    // 정규화 통계 파일도 함께 다운로드
    let base_name = target_model_name.trim_end_matches(".mpk");
    let stats_name = format!("norm_stats_{}.json", base_name);
    let stats_url = format!("{}/api/download/{}", server_url, stats_name);
    if let Ok(mut s_res) = client.get(&stats_url).send() {
        if s_res.status().is_success() {
            if let Ok(mut dest_s) = File::create(&stats_name) {
                let _ = io::copy(&mut s_res, &mut dest_s);
                println!("✔ 정규화 통계 파일 동시 수신 완료: {}", stats_name);
            }
        }
    }

    println!("\n이제 [3. 실시간 추론] 메뉴에서 이 모델을 바로 사용할 수 있습니다!");
    Ok(())
}
