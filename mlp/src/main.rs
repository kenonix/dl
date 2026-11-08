mod dataset;
mod model;

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, Tensor},
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};
use dataset::{EmgBatcher, EmgDataset, NormStats};
use model::{Model, ModelConfig};
use rand::seq::SliceRandom;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::{Duration, Instant};

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

const CSV_FILE: &str = "emg_dataset_20260921_214355.csv";
const ARTIFACTS_DIR: &str = "./model_artifacts";
const MODEL_SAVE_PATH: &str = "./model_artifacts/final_model";
const STATS_SAVE_PATH: &str = "./model_artifacts/norm_stats.json";

fn get_action_name(label: usize) -> &'static str {
    match label {
        0 => "휴식 (Relax)",
        1 => "주먹 쥐기 (Fist)",
        2 => "손가락 펴기 (Open)",
        3 => "엄지 굽히기 (Thumb Flex)",
        _ => "알 수 없음 (Unknown)",
    }
}

// 1. 학습 파이프라인
fn run_training() {
    println!("\n[*] 데이터 로드 및 전처리를 시작합니다...");
    let device = WgpuDevice::default();

    if !Path::new(ARTIFACTS_DIR).exists() {
        std::fs::create_dir_all(ARTIFACTS_DIR).expect("디렉토리 생성 실패");
    }

    let (train_dataset, valid_dataset, stats) = EmgDataset::prepare_data(CSV_FILE);
    stats.save(STATS_SAVE_PATH);
    println!("[+] Z-Score 정규화 파라미터 저장 완료 -> {}", STATS_SAVE_PATH);

    let batcher_train = EmgBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = EmgBatcher::<MyBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(32)
        .shuffle(42)
        .num_workers(1)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(32)
        .num_workers(1)
        .build(valid_dataset);

    let config = ModelConfig::new();
    let model = config.init::<MyAutodiffBackend>(&device);

    println!("[*] 모델 학습을 시작합니다 (20 Epochs)...");
    let learner = LearnerBuilder::new(ARTIFACTS_DIR)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(20)
        .build(model, AdamConfig::new().init(), 1e-3);

    let trained_model = learner.fit(dataloader_train, dataloader_valid);

    let recorder = CompactRecorder::new();
    recorder
        .record(trained_model.into_record(), MODEL_SAVE_PATH.into())
        .expect("모델 저장 실패");

    println!("[+] 최종 모델 가중치 저장 완료 -> {}", MODEL_SAVE_PATH);
    println!("[+] 학습이 완료되었습니다!\n");
}

// 2. 모델 로드 함수
fn load_model_for_inference(device: &WgpuDevice) -> Option<Model<MyBackend>> {
    let model_path = format!("{}.mpk", MODEL_SAVE_PATH);
    if !Path::new(&model_path).exists() && !Path::new(MODEL_SAVE_PATH).exists() {
        println!("[-] 학습된 모델 파일이 존재하지 않습니다. 먼저 1번을 실행하세요.");
        return None;
    }

    let recorder = CompactRecorder::new();
    let record = recorder
        .load(MODEL_SAVE_PATH.into(), device)
        .expect("가중치 로드 실패");

    let model = ModelConfig::new().init::<MyBackend>(device).load_record(record);
    Some(model)
}

// 3. 정적 데이터셋 추론 테스트
fn run_inference() {
    let device = WgpuDevice::default();
    let model = match load_model_for_inference(&device) {
        Some(m) => m,
        None => return,
    };

    if !Path::new(STATS_SAVE_PATH).exists() {
        println!("[-] 정규화 파일이 없습니다. 먼저 학습을 진행하세요.");
        return;
    }
    let stats = NormStats::load(STATS_SAVE_PATH);
    let raw_samples = EmgDataset::load_raw(CSV_FILE);

    println!("\n=== 정적 데이터셋 추론 테스트 ===");
    let mut rng = rand::thread_rng();
    let sample_choices: Vec<_> = raw_samples.choose_multiple(&mut rng, 5).cloned().collect();

    for (idx, sample) in sample_choices.iter().enumerate() {
        let start_time = Instant::now();
        let norm_features = stats.normalize(&sample.features);
        let input = Tensor::<MyBackend, 1>::from_floats(norm_features.as_slice(), &device)
            .reshape([1, 75]);

        let logits = model.forward(input);
        let probs = softmax(logits.clone(), 1);
        let pred_label: i32 = logits.argmax(1).into_scalar();
        let latency = start_time.elapsed();
        let probs_vec: Vec<f32> = probs.into_data().to_vec().unwrap();

        println!(
            "샘플 #{}: 정답: [{}] -> 예측: [{}] (소요: {:?})",
            idx + 1,
            get_action_name(sample.label),
            get_action_name(pred_label as usize),
            latency
        );
        println!(
            "   확률: Relax: {:.1}%, Fist: {:.1}%, Open: {:.1}%, Thumb: {:.1}%",
            probs_vec[0] * 100.0,
            probs_vec[1] * 100.0,
            probs_vec[2] * 100.0,
            probs_vec[3] * 100.0
        );
    }
}

// 4. 실시간 시리얼 포트(/dev/ttyACM0) 스트리밍 추론 모드
fn run_serial_inference() {
    let device = WgpuDevice::default();
    let model = match load_model_for_inference(&device) {
        Some(m) => m,
        None => return,
    };

    if !Path::new(STATS_SAVE_PATH).exists() {
        println!("[-] 정규화 파일이 없습니다. 먼저 학습을 완료하세요.");
        return;
    }
    let stats = NormStats::load(STATS_SAVE_PATH);

    // 시리얼 포트 경로 설정 (/dev/ttyACM0 우선 확인)
    let default_port = if Path::new("/dev/ttyACM0").exists() {
        "/dev/ttyACM0"
    } else if Path::new("/dev/ACM0").exists() {
        "/dev/ACM0"
    } else {
        "/dev/ttyACM0"
    };

    print!("\n시리얼 포트 경로 입력 (기본값: {}): ", default_port);
    io::stdout().flush().unwrap();
    let mut port_input = String::new();
    io::stdin().read_line(&mut port_input).unwrap();
    let port_path = match port_input.trim() {
        "" => default_port,
        custom => custom,
    };

    print!("통신 보드레이트(Baud rate) 입력 (기본값: 115200): ");
    io::stdout().flush().unwrap();
    let mut baud_input = String::new();
    io::stdin().read_line(&mut baud_input).unwrap();
    let baud_rate: u32 = baud_input.trim().parse().unwrap_or(115200);

    println!("[*] 포트 {} 연결 시도 중 (Baud: {})...", port_path, baud_rate);

    let port = match serialport::new(port_path, baud_rate)
        .timeout(Duration::from_millis(500))
        .open()
    {
        Ok(p) => p,
        Err(e) => {
            println!("[-] 시리얼 포트 열기 실패: {}", e);
            println!("    (권한 문제일 경우: 'sudo usermod -a -G dialout $USER' 후 재로그인 필요)");
            return;
        }
    };

    println!("[+] 시리얼 포트 연결 성공! 실시간 데이터 스트리밍 추론을 시작합니다.");
    println!("[*] 스트림을 중단하려면 Ctrl+C를 누르세요.\n");
    println!("{:-<70}", "");
    println!("{:<24} | {:<10} | {:<12}", "인식된 제스처", "신뢰도", "추론 지연시간");
    println!("{:-<70}", "");

    let mut reader = BufReader::new(port);
    let mut line_buf = String::new();

    loop {
        line_buf.clear();
        match reader.read_line(&mut line_buf) {
            Ok(0) => {
                // 스트림 종료 또는 타임아웃
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
            Ok(_) => {
                let trimmed = line_buf.trim();
                if trimmed.is_empty() || trimmed.starts_with('A') || trimmed.starts_with('#') {
                    // 헤더 또는 빈 줄 무시
                    continue;
                }

                // CSV 라인에서 75개 부동소수점 파싱
                let features: Vec<f32> = trimmed
                    .split(',')
                    .filter_map(|s| s.trim().parse::<f32>().ok())
                    .collect();

                // 75차원 특징 벡터 검증 (정확히 75개이거나 라벨 포함 76~77개일 경우 앞의 75개 슬라이싱)
                if features.len() < 75 {
                    continue;
                }
                let input_features = &features[..75];

                let start_time = Instant::now();

                // 1. 저장된 통계로 Z-Score 표준화
                let norm_features = stats.normalize(input_features);

                // 2. 텐서 생성 [1, 75]
                let input_tensor = Tensor::<MyBackend, 1>::from_floats(norm_features.as_slice(), &device)
                    .reshape([1, 75]);

                // 3. 모델 순전파 추론
                let logits = model.forward(input_tensor);
                let probs = softmax(logits.clone(), 1);
                let pred_label: i32 = logits.argmax(1).into_scalar();
                let latency = start_time.elapsed();

                let probs_vec: Vec<f32> = probs.into_data().to_vec().unwrap();
                let confidence = probs_vec[pred_label as usize] * 100.0;

                // 출력 포맷팅
                print!(
                    "\r\x1b[K{:<24} | {:>8.2}% | {:>8.2?}",
                    get_action_name(pred_label as usize),
                    confidence,
                    latency
                );
                io::stdout().flush().unwrap();
            }
            Err(ref e) if e.kind() == io::ErrorKind::TimedOut => {
                // 타임아웃 발생 시 대기
                continue;
            }
            Err(e) => {
                println!("\n[-] 시리얼 읽기 오류 발생: {}", e);
                break;
            }
        }
    }
}

// 5. 메인 콘솔 UI 루프
fn main() {
    loop {
        println!("\n=======================================================");
        println!("       sEMG 5채널 제스처 인식 시스템 (Burn 엔진)");
        println!("=======================================================");
        println!("  1. 데이터 전처리 및 신경망 학습 (Train)");
        println!("  2. 정적 데이터셋 검증 테스트 (Inference Test)");
        println!("  3. 실시간 시리얼(/dev/ttyACM0) 추론 모드 (Live Stream)");
        println!("  4. 프로그램 종료 (Exit)");
        println!("=======================================================");
        print!("메뉴 번호를 입력하세요 > ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim() {
            "1" => run_training(),
            "2" => run_inference(),
            "3" => run_serial_inference(),
            "4" => {
                println!("프로그램을 종료합니다.");
                break;
            }
            _ => println!("잘못된 입력입니다. 1 ~ 4번 중에서 선택해 주세요."),
        }
    }
}
