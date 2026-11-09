use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::Tensor;
use burn::backend::wgpu::WgpuDevice;
use emg_ml_pipeline::{
    fft::{
        EmgFftProcessor, NUM_CHANNELS, TD_CORE_FEATURES, TOTAL_FEATURES_ALL,
        TOTAL_FEATURES_WITH_RAW,
    },
    models::{
        EmgCnnLstmModel, EmgCnnModel, EmgLstmModel, EmgMlpConfig, EmgMlpModel,
        EmgModularMlpModel, EmgTcnModel, ModelArchitecture, RealTimeRingBuffer,
        MODULAR_MLP_WINDOW_SIZE,
    },
    train::{
        load_mlp_dataset, load_modular_mlp_dataset, train_model, MyBackend, NormalizationStats,
        TrainingConfig, TrainingProgress, NUM_CLASSES, SEQ_LEN,
    },
};
use glob::glob;
use rand::seq::SliceRandom;
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
        println!(" 🦾 EMG 고성능 딥러닝 & 지표 추출 파이프라인 (Burn 0.16)");
        println!("========================================================");
        println!(" 1. 📥 데이터 수집 (원시값 + FFT + 시간도메인 75차원 동시 저장)");
        println!(" 2. 🧠 로컬 모델 학습 (MLP / 🚀Block-MLP(SwiGLU) / 1D-TCN / 1D-CNN / LSTM / CNN+LSTM)");
        println!(" 3. 🚀 실시간 추론 (MLP / 🚀Block-MLP 링버퍼 추론 & 복합 시계열 모델)");
        println!(" 4. 📊 정적 데이터셋 검증 테스트 (Inference Test - MLP & 🚀Block-MLP 고정밀 검증)");
        println!(" 5. 📈 3차원 데이터셋 시각화 (Plotly 3D 그래프 - PCA 공간 & 생체 물리 지표)");
        println!(" 6. 🌐 서버 원격 학습 및 모델 다운로드 (순수 Rust 서버 연동)");
        println!(" 7. 🔬 3초 정제 4D LDA & SVM 분류기 분석 및 학습 (Pure Rust)");
        println!(" 8. 🩺 센서 신호 실시간 진단 & 접촉 테스트 (Live Health Check)");
        println!(" 9. 📊 데이터셋 종합 품질 분석기 (Quality Assessor)");
        println!(" 10. 🚪 프로그램 종료");
        print!("메뉴를 선택하세요 (1-10) > ");
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
                run_static_dataset_test(&device);
            }
            "5" => {
                emg_ml_pipeline::run_3d_visualization_menu();
            }
            "6" => {
                if let Err(e) = run_remote_server_pipeline() {
                    eprintln!("❌ 서버 연동 에러: {}", e);
                }
            }
            "7" => {
                if let Err(e) = emg_ml_pipeline::lda::run_lda_svm_analysis_interactive() {
                    eprintln!("❌ LDA & SVM 분석 에러: {}", e);
                }
            }
            "8" => {
                if let Err(e) = run_standalone_sensor_diagnostic() {
                    eprintln!("❌ 센서 진단 에러: {}", e);
                }
            }
            "9" => {
                if let Err(e) = emg_ml_pipeline::run_dataset_quality_menu() {
                    eprintln!("❌ 데이터 품질 분석 에러: {}", e);
                }
            }
            "10" => {
                println!("프로그램을 종료합니다.");
                break;
            }
            _ => {
                println!("❌ 올바른 번호를 입력하세요 (1~10)");
            }
        }

    }
}

// =========================================================================
// 🩺 센서 신호 실시간 진단 및 전극 접촉 검사 (Health Check Engine)
// =========================================================================
fn run_sensor_diagnostic_on_stream<R: BufRead>(reader: &mut R) -> io::Result<bool> {
    println!("\n========================================================");
    println!(" 🩺 [센서 접촉 및 신호 품질 실시간 진단 모드]");
    println!("========================================================");
    println!(" 💡 팔에 힘을 뺐을 때(휴식): RMS < 100, 원시값 안정 (✔️ 정상)");
    println!(" 💡 주먹을 쥐었을 때(수축): RMS 50~200 이상 힘에 비례 상승 확인");
    println!(" ⚠️  이상(전극 들뜸/GND 탈착): 1023 포화 또는 RMS > 300 폭증 (❌ 위험)");
    println!("--------------------------------------------------------");
    println!("👉 신호가 안정적인 것을 확인한 후, [엔터 키]를 누르면 다음 단계로 진행합니다.");
    println!("👉 (취소하고 메뉴로 돌아가려면 'q' 입력 후 엔터)\n");

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut s = String::new();
        let _ = io::stdin().read_line(&mut s);
        let _ = tx.send(s.trim().to_string());
    });

    let mut processor = EmgFftProcessor::new();
    let mut ema_rms = 0.0f32;
    let mut last_raw = [0.0f32; NUM_CHANNELS];
    let mut sample_count = 0usize;

    let user_input = loop {
        if let Ok(input) = rx.try_recv() {
            break input;
        }

        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
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
                    sample_count += 1;
                    last_raw.copy_from_slice(&parts);
                    let feats75 = processor.process_sample_all(&last_raw);
                    let total_rms = EmgFftProcessor::extract_total_ac_rms(&feats75);
                    if sample_count <= 5 {
                        ema_rms = total_rms;
                    } else {
                        ema_rms = 0.20 * total_rms + 0.80 * ema_rms;
                    }

                    let mut saturated_channels = Vec::new();
                    for ch in 0..NUM_CHANNELS {
                        if last_raw[ch] >= 1020.0 {
                            saturated_channels.push(ch);
                        }
                    }

                    let status_desc = if !saturated_channels.is_empty() {
                        format!("❌ [위험: Ch{:?} 1023 포화/들뜸!]", saturated_channels)
                    } else if ema_rms > 300.0 {
                        "❌ [위험: RMS 300+ 노이즈 폭증! 전극/GND 확인]".to_string()
                    } else if ema_rms > 120.0 {
                        "⚠️ [주의: RMS 다소 높음 / 힘을 완전히 뺐는지 확인]".to_string()
                    } else {
                        "✔️ [양호: 정상 접촉 및 안정 신호]".to_string()
                    };

                    print!(
                        "\r🩺 [RAW: {:4.0} {:4.0} {:4.0} {:4.0} {:4.0}] | RMS:{:5.1} | {} | [엔터: 진행] ",
                        last_raw[0], last_raw[1], last_raw[2], last_raw[3], last_raw[4],
                        ema_rms,
                        status_desc
                    );
                    io::stdout().flush()?;
                }
            }
            Err(_) => {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    };
    println!();

    if user_input.eq_ignore_ascii_case("q") {
        println!("🛑 진단 화면에서 사용자에 의해 취소되었습니다.");
        return Ok(false);
    }

    if ema_rms > 300.0 {
        println!("\n⚠️ [심각한 경고] 현재 센서 RMS 노이즈({:.1})가 매우 심각한 수준입니다!", ema_rms);
        println!("   이 상태로 데이터를 수집하면 휴식과 손 쥐기가 전혀 구분되지 않아 모델이 학습되지 않습니다.");
        println!("   💡 조치 요령:");
        println!("      1. 기준 전극(Reference/GND)이 팔꿈치 뼈 부위에 단단히 밀착되어 있는지 확인하세요.");
        println!("      2. 전극 패드의 젤이 마르지 않았는지, 피부에 잘 붙었는지 확인하세요.");
        println!("      3. 아두이노 GND/신호 점퍼선 연결 상태를 확인하세요.");
        print!("\n   그래도 강제로 수집을 계속 진행하시겠습니까? (y/N) > ");
        io::stdout().flush()?;
        let mut ans = String::new();
        io::stdin().read_line(&mut ans)?;
        if !ans.trim().eq_ignore_ascii_case("y") {
            println!("❌ 데이터 수집을 중단합니다. 센서 접촉을 해결한 후 다시 시도해주세요.");
            return Ok(false);
        }
    } else {
        println!("✔ 센서 접촉 진단 통과 (현재 RMS: {:.1})! 수집을 시작합니다.\n", ema_rms);
    }

    Ok(true)
}

fn run_standalone_sensor_diagnostic() -> io::Result<()> {
    let port_name = "/dev/ttyACM0";
    let baud_rate = 9600;

    println!("\n시리얼 포트 연결 중: {}...", port_name);
    let port = serialport::new(port_name, baud_rate)
        .timeout(Duration::from_millis(500))
        .open()
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e))?;

    let mut reader = BufReader::new(port);
    let _ = run_sensor_diagnostic_on_stream(&mut reader)?;
    Ok(())
}

struct SetFrameRecord {
    row_str: String,
    label: usize,
    total_rms: f32,
    is_sat: bool,
    is_silent: bool,
}

// =========================================================================
// 1. 센서 원시값 + FFT 스펙트럼 + 확장 시간 도메인 6대 지표(총 75차원) 동시 수집
// =========================================================================
fn run_data_collection() -> io::Result<()> {
    println!("\n========================================================");
    println!(" 📥 EMG 데이터 수집 모드 선택");
    println!("========================================================");
    println!(" [1] ✊🛡️ 잡음 내성 3대 동작 + 지속 수축 (휴식 ➔ 팔 흔들기/잡음 ➔ 주먹 유지) [🥇 강력 추천]");
    println!(" [2] ✊ 2대 기본 동작 수집 (휴식 ➔ 손 쥐기 및 지속 유지)");
    println!(" [3] ✊🖐️ 4대 핵심 제스처 수집 (휴식 ➔ 주먹유지 ➔ 손펴기 ➔ 엄지)");
    println!(" [4] 🖐️ 다섯 손가락 개별 동작 수집 (휴식 ➔ 엄지 ➔ 검지 ➔ 중지 ➔ 약지 ➔ 새끼)");
    print!("선택 번호 (1-4, 기본 1) > ");
    io::stdout().flush()?;

    let mut mode_input = String::new();
    io::stdin().read_line(&mut mode_input)?;

    let choice = mode_input.trim();
    let (mode_name, file_prefix, cycle_steps, guide_msg) = if choice == "2" {
        (
            "✊ 2대 기본 동작 (휴식 ➔ 손 쥐기 및 유지)",
            "emg_dataset_fist",
            vec![
                ActionStep { name: "휴식 (Relax)", label_id: 0, duration_secs: 8 },
                ActionStep { name: "손 쥐기 및 지속 유지 (Sustained Fist)", label_id: 1, duration_secs: 11 },
            ],
            "=== 💡 2대 기본 동작 + 지속 수축(Sustained Hold) 가이드 ===\n\
             - [1단계: 휴식] 팔에 완전히 힘을 빼고 편안히 유지하세요. (8초)\n\
             - [2단계: 손 쥐기 및 유지] 주먹을 꽉 쥐고 끝날 때까지 힘을 풀지 말고 유지하세요! (11초)\n\
             - 시작 직후 2초 동안은 움직임 과도기 노이즈 방지를 위해 자동 제외됩니다.",
        )
    } else if choice == "3" {
        (
            "✊🖐️ 4대 핵심 제스처",
            "emg_dataset",
            vec![
                ActionStep { name: "휴식 (Relax)", label_id: 0, duration_secs: 8 },
                ActionStep { name: "주먹 쥐기 및 지속 유지 (Sustained Fist)", label_id: 1, duration_secs: 11 },
                ActionStep { name: "손가락 펴기 (Open)", label_id: 2, duration_secs: 8 },
                ActionStep { name: "엄지 굽히기 (Thumb Flex)", label_id: 3, duration_secs: 8 },
            ],
            "=== 💡 4대 제스처 + 지속 수축 가이드 ===\n\
             - 4대 동작 [휴식 ➔ 주먹유지 ➔ 손펴기 ➔ 엄지]을 순환 수집합니다.\n\
             - [주먹 쥐기] 시에는 주먹을 쥐고 11초 동안 끝까지 풀지 말고 유지하세요!\n\
             - 카운트다운이 시작되기 전에 미리 해당 자세를 취하고 계세요!",
        )
    } else if choice == "4" {
        (
            "🖐️ 다섯 손가락 개별 동작",
            "emg_dataset_fingers",
            vec![
                ActionStep { name: "휴식 (Relax)", label_id: 0, duration_secs: 8 },
                ActionStep { name: "엄지 굽히기 (Thumb Flex)", label_id: 1, duration_secs: 8 },
                ActionStep { name: "검지 굽히기 (Index Flex)", label_id: 2, duration_secs: 8 },
                ActionStep { name: "중지 굽히기 (Middle Flex)", label_id: 3, duration_secs: 8 },
                ActionStep { name: "약지 굽히기 (Ring Flex)", label_id: 4, duration_secs: 8 },
                ActionStep { name: "새끼 굽히기 (Pinky Flex)", label_id: 5, duration_secs: 8 },
            ],
            "=== 💡 다섯 손가락 데이터 수집 가이드 ===\n\
             - 6대 동작 [휴식 ➔ 엄지 ➔ 검지 ➔ 중지 ➔ 약지 ➔ 새끼]을 순환 수집합니다.\n\
             - 시작 직후 2초 동안은 움직임 과도기 노이즈 방지를 위해 자동 제외됩니다.",
        )
    } else {
        (
            "✊🛡️ 잡음 내성 3대 동작 + 지속 수축 (휴식 ➔ 팔 흔들기 ➔ 주먹 지속 유지)",
            "emg_dataset_noise_robust",
            vec![
                ActionStep { name: "정적 휴식 (Rest)", label_id: 0, duration_secs: 8 },
                ActionStep { name: "동적 잡음 (Noise / Arm Motion)", label_id: 1, duration_secs: 8 },
                ActionStep { name: "진짜 주먹 및 지속 유지 (Sustained Fist)", label_id: 2, duration_secs: 11 },
            ],
            "=== 💡 잡음 내성 3대 동작 + 지속 수축(Sustained Hold) 수집 가이드 ===\n\
             - [1단계: 정적 휴식] 팔에 힘을 완전히 빼고 편안히 가만히 유지하세요. (8초)\n\
             - [2단계: 동적 잡음] 일상적인 팔 흔들기, 손목 돌리기를 자연스럽게 수행하세요! (8초, 주먹은 쥐지 마세요)\n\
             - [3단계: 주먹 지속 유지] 카운트다운 후 주먹을 꽉 쥐고, **끝날 때까지 힘을 풀지 말고 계속 쥐고 유지**하세요! (11초)\n\
             - 🚀 [피로 감쇠 패턴 학습] 주먹을 길게 유지할 때 진폭이 떨어지는 생리적 피로 패턴까지 AI가 학습합니다!\n\
             - 시작 직후 2초 동안은 과도기 노이즈 방지를 위해 버퍼 갱신 후 순수 데이터만 기록됩니다.",
        )
    };

    println!("--------------------------------------------------");
    println!(" ⚙️ [녹화 세트(사이클) 설정]");
    print!(" 녹화할 세트 수를 입력하세요 (기본: 6세트, Enter 누르면 6) > ");
    io::stdout().flush()?;
    let mut cycles_input = String::new();
    io::stdin().read_line(&mut cycles_input)?;
    let total_cycles: usize = cycles_input.trim().parse::<usize>().unwrap_or(6).max(1);
    println!(" ✔ 총 {}세트 순환 녹화로 설정되었습니다.\n", total_cycles);

    let port_name = "/dev/ttyACM0";
    let baud_rate = 9600;

    println!("\n시리얼 포트 연결 중: {}...", port_name);
    let port = serialport::new(port_name, baud_rate)
        .timeout(Duration::from_millis(1000))
        .open()
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e))?;

    let mut reader = BufReader::new(port);

    // 🚀 [필수 1단계: 실시간 센서 접촉 및 노이즈 진단]
    let proceed = run_sensor_diagnostic_on_stream(&mut reader)?;
    if !proceed {
        return Ok(());
    }

    let file_name = format!(
        "{}_{}.csv",
        file_prefix,
        chrono::Local::now().format("%Y%m%d_%H%M%S")
    );
    let mut csv_file = File::create(&file_name)?;

    // 75차원 전체 특징 CSV 헤더 작성 (원시 5 + FFT 40 + 시간 도메인 30 = 75)
    writeln!(csv_file, "{}", EmgFftProcessor::csv_header())?;
    println!("저장 파일 생성: {} (특징 총 75차원: 원시 5 + FFT 40 + 시간 도메인 30)\n", file_name);

    println!("{}", guide_msg);
    println!("준비되셨으면 엔터 키를 누르세요...");
    let mut dummy = String::new();
    io::stdin().read_line(&mut dummy)?;

    let mut processor = EmgFftProcessor::new();

    let mut completed_cycles = 0;
    while completed_cycles < total_cycles {
        let current_cycle = completed_cycles + 1;
        println!("\n==================================================");
        println!(" 🔄 [세트 {} / {} 시작] {} 순환 측정", current_cycle, total_cycles, mode_name);
        println!("==================================================");

        let mut set_frames: Vec<SetFrameRecord> = Vec::new();

        for (idx, step) in cycle_steps.iter().enumerate() {
            let global_step = completed_cycles * cycle_steps.len() + idx + 1;
            let total_steps = total_cycles * cycle_steps.len();
            println!("\n--------------------------------------------------");
            println!("[ 세트 {}/{} | 총 진행 {}/{} 단계 ]", current_cycle, total_cycles, global_step, total_steps);
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

                    let parts: Vec<f32> = trimmed
                        .split(',')
                        .filter_map(|s| s.trim().parse::<f32>().ok())
                        .collect();

                    if parts.len() == NUM_CHANNELS {
                        let mut raw_arr = [0.0f32; NUM_CHANNELS];
                        raw_arr.copy_from_slice(&parts[..NUM_CHANNELS]);

                        if start_time.elapsed() < transition_buffer {
                            processor.update_buffer(&raw_arr);
                            print!("\r⏳ [과도기 대기] 자세 잡는 중... [{}]     ", trimmed);
                            io::stdout().flush()?;
                            continue;
                        }

                        if !is_recording_started {
                            println!("\n✔ [데이터 수집 중] 세트 {} - 75차원 지표 기록 중!", current_cycle);
                            is_recording_started = true;
                        }

                        let all_features = processor.process_sample_all(&raw_arr);

                        let mut row_str = String::new();
                        for (i, val) in all_features.iter().enumerate() {
                            if i > 0 {
                                row_str.push(',');
                            }
                            row_str.push_str(&format!("{:.2}", val));
                        }
                        row_str.push_str(&format!(",{},{}\n", step.label_id, step.name));

                        let total_rms = EmgFftProcessor::extract_total_ac_rms(&all_features);
                        let is_sat = raw_arr.iter().any(|&x| x >= 1020.0);
                        let is_silent = total_rms < 10.0;

                        set_frames.push(SetFrameRecord {
                            row_str,
                            label: step.label_id,
                            total_rms,
                            is_sat,
                            is_silent,
                        });

                        let elapsed_f = start_time.elapsed().as_secs_f32();
                        let remaining_secs = (total_duration.as_secs_f32() - elapsed_f).max(0.0);
                        let short_name = step.name.split_whitespace().next().unwrap_or(step.name);

                        let phase_cue = if step.label_id == 2 || (choice == "2" && step.label_id == 1) {
                            if elapsed_f < 4.0 {
                                "✊ [초기 수축] 꽉 쥐세요!"
                            } else if elapsed_f < 7.5 {
                                "✊ [지속 유지] 힘 풀지 말고 계속 쥐고 계세요!"
                            } else {
                                "✊ [피로 패턴 학습] 진폭이 줄어도 끝까지 유지!"
                            }
                        } else if step.label_id == 0 {
                            "💤 [완전 휴식] 팔에 힘을 완전히 빼세요..."
                        } else {
                            "🌊 [동적 잡음] 일상적 팔 흔들기/손목 돌리기..."
                        };

                        print!(
                            "\r[기록 중] 세트 {}/{} | {:<6} (남은시간:{:3.1}s) | RMS:{:5.1} | 원시:[{:.0},{:.0},{:.0},{:.0},{:.0}] | {}  ",
                            current_cycle, total_cycles, short_name, remaining_secs, total_rms, raw_arr[0], raw_arr[1], raw_arr[2], raw_arr[3], raw_arr[4], phase_cue
                        );
                        io::stdout().flush()?;
                    }
                }
            }
            println!("\n✔ [세트 {}] {} 수집 완료!", current_cycle, step.name);
        }

        // =========================================================================
        // 🔍 [세트 무결성 검증 보고서 및 재녹화 확인]
        // =========================================================================
        println!("\n==================================================");
        println!(" 🔍 [세트 {} / {} 데이터 무결성 검증 분석]", current_cycle, total_cycles);
        println!("==================================================");

        let total_frames = set_frames.len();
        let rest_frames: Vec<&SetFrameRecord> = set_frames.iter().filter(|f| f.label == 0).collect();
        let fist_frames: Vec<&SetFrameRecord> = set_frames.iter().filter(|f| {
            if choice == "2" { f.label == 1 } else { f.label == 2 }
        }).collect();

        let rest_rms = if !rest_frames.is_empty() {
            rest_frames.iter().map(|f| f.total_rms).sum::<f32>() / rest_frames.len() as f32
        } else { 0.0 };

        let fist_rms = if !fist_frames.is_empty() {
            fist_frames.iter().map(|f| f.total_rms).sum::<f32>() / fist_frames.len() as f32
        } else { 0.0 };

        let (fist_early_rms, fist_late_rms) = if fist_frames.len() >= 10 {
            let half = fist_frames.len() / 2;
            let early = fist_frames[..half].iter().map(|f| f.total_rms).sum::<f32>() / half as f32;
            let late = fist_frames[half..].iter().map(|f| f.total_rms).sum::<f32>() / (fist_frames.len() - half) as f32;
            (early, late)
        } else {
            (fist_rms, fist_rms)
        };

        let contrast_ratio = if rest_rms > 1e-3 { fist_rms / rest_rms } else { 1.0 };
        let sat_count = set_frames.iter().filter(|f| f.is_sat).count();
        let silent_count = set_frames.iter().filter(|f| f.is_silent).count();

        println!(" 📊 [세트 {} 품질 지표 보고서]", current_cycle);
        println!("  - 총 수집 프레임: {} 개 ({:.1}초 분량)", total_frames, total_frames as f32 / 50.0);
        println!("  - 휴식기(Rest) 평균 RMS: {:.1}", rest_rms);
        println!("  - 주먹기(Fist) 평균 RMS: {:.1} (초기: {:.1} ➔ 후반 유지: {:.1})", fist_rms, fist_early_rms, fist_late_rms);
        println!("  - 수축 대비비 (Fist / Rest): x{:.2}배", contrast_ratio);
        println!("  - 센서 포화(들뜸) 프레임: {} 개", sat_count);
        println!("  - 센서 침묵(단선) 프레임: {} 개", silent_count);

        if contrast_ratio >= 1.6 && sat_count == 0 && silent_count == 0 {
            println!("  - 종합 신호 품질: 🟢 [우수] 휴식과 주먹의 대비가 우수하며 신호가 매우 깨끗합니다.");
        } else if contrast_ratio >= 1.2 && sat_count < 25 {
            println!("  - 종합 신호 품질: 🟡 [보통] 사용 가능하지만 다소 잡음이 있거나 대비가 낮습니다.");
        } else {
            println!("  - 종합 신호 품질: 🔴 [주의] 주먹과 휴식 신호 구분이 어렵거나 센서가 들떴습니다! (재녹화 강력 권장)");
        }
        println!("--------------------------------------------------");
        print!(" 👉 세트 {} 데이터를 저장하고 계속 진행하시겠습니까? (Enter/y: 저장, n: 폐기 후 재녹화) > ", current_cycle);
        io::stdout().flush()?;

        let mut dec_input = String::new();
        io::stdin().read_line(&mut dec_input)?;
        let dec = dec_input.trim().to_lowercase();
        if dec == "n" || dec == "no" {
            println!("\n🗑️ [세트 {} 폐기] 방금 수집된 세트를 버리고 세트 {}을 처음부터 다시 수집합니다.\n", current_cycle, current_cycle);
            continue;
        }

        // CSV에 영구 기록
        for f in &set_frames {
            csv_file.write_all(f.row_str.as_bytes())?;
        }
        csv_file.flush()?;
        completed_cycles += 1;
        println!(" ✔ [세트 {} 영구 저장 완료] 누적 완료: {} / {} 세트", current_cycle, completed_cycles, total_cycles);

        if completed_cycles < total_cycles {
            println!("\n☕ [세트 {}/{} 완료!] 다음 세트 전 3초간 팔에 힘을 빼고 편안히 휴식하세요...", current_cycle, total_cycles);
            for i in (1..=3).rev() {
                print!("{}... ", i);
                io::stdout().flush()?;
                std::thread::sleep(Duration::from_secs(1));
            }
            println!();
        }
    }

    println!("\n🎉 총 {}세트 고품질 다차원(75차원) {} 데이터 수집 완료! 파일: {}", total_cycles, mode_name, file_name);

    // 🚀 [신규] 35D 경량 시간도메인 전용 데이터셋 동시 자동 생성 (FFT 제외 경량화 버전)
    let file_35d_name = file_name.replace(".csv", "_35d.csv");
    if let (Ok(in_f), Ok(out_f)) = (File::open(&file_name), File::create(&file_35d_name)) {
        use std::io::BufWriter;
        let mut rdr = BufReader::new(in_f);
        let mut wrt = BufWriter::new(out_f);
        let mut line_buf = String::new();
        if rdr.read_line(&mut line_buf).is_ok() {
            let _ = writeln!(wrt, "{}", EmgFftProcessor::csv_header_td35());
        }
        line_buf.clear();
        while let Ok(n) = rdr.read_line(&mut line_buf) {
            if n == 0 { break; }
            let trimmed = line_buf.trim();
            if !trimmed.is_empty() {
                let parts: Vec<&str> = trimmed.split(',').collect();
                if parts.len() >= 77 {
                    let mut row_35 = Vec::with_capacity(37);
                    row_35.extend_from_slice(&parts[0..5]);   // Raw 5
                    row_35.extend_from_slice(&parts[45..75]); // TD 30
                    row_35.extend_from_slice(&parts[75..77]); // label, action_name
                    let _ = writeln!(wrt, "{}", row_35.join(","));
                }
            }
            line_buf.clear();
        }
        println!("✨ [자동 동기화] 35D 경량 시간도메인 전용 데이터셋 자동 생성 완료: {}", file_35d_name);
    }

    // 🚀 [신규] 수집 완료 즉시 데이터셋 종합 품질 검증 및 성적표 브리핑
    println!("\n📊 수집된 데이터셋의 종합 품질 검증을 시작합니다...");
    let _ = emg_ml_pipeline::quality::evaluate_and_prompt_after_collection(&file_name)?;

    Ok(())
}

// =========================================================================
// 2. 로컬 모델 학습 (1D-TCN / 1D-CNN / LSTM / CNN+LSTM)
// =========================================================================
fn run_local_training(device: &WgpuDevice) -> io::Result<()> {
    let mut found_files = Vec::new();
    for entry in glob("emg_dataset_*.csv").expect("Failed glob") {
        if let Ok(path) = entry {
            if std::fs::metadata(&path).map(|m| m.len() > 600).unwrap_or(false) {
                found_files.push(path);
            }
        }
    }
    for entry in glob("uploads/*.csv").expect("Failed glob") {
        if let Ok(path) = entry {
            if std::fs::metadata(&path).map(|m| m.len() > 600).unwrap_or(false) {
                found_files.push(path);
            }
        }
    }

    found_files.sort();
    if found_files.is_empty() {
        println!("\n❌ [에러] 수집된 데이터셋 파일이 없습니다. 1번 데이터 수집을 먼저 실행해주세요.");
        return Ok(());
    }

    println!("\n[데이터셋 목록]");
    for (i, file) in found_files.iter().enumerate() {
        let file_str = file.display().to_string();
        let tag = if file_str.contains("merged_35d") {
            "  <-- 🌟 [🥇 1순위 추천! 35D 경량 시간도메인 전용 - FFT 제외 고속·초안정성 (총 4,347 샘플)]"
        } else if file_str.contains("merged") {
            "  <-- 🌟 [추천! 10/02 + 11/08 멀티세션 75D 통합 데이터셋 (총 4,347 샘플)]"
        } else if file_str.contains("lda4d") {
            "  <-- 🌟 [2번 아키텍처: 4D LDA 추천 / 5ch ➔ 75D ➔ 4D 투영 ➔ 딥러닝 5종]"
        } else if file_str.contains("clean") {
            "  <-- [75차원 DC-Invariant FFT 클린 데이터셋]"
        } else {
            ""
        };
        println!("  [{}] {}{}", i + 1, file.display(), tag);
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

    println!("\n[학습할 모델 아키텍처 선택 (단일 또는 쉼표/공백으로 복수 선택 가능)]");
    println!("  [1] MLP (75D/4D 기본 5층 완전연결망 - 단일 프레임 즉시 추론)");
    println!("  [2] 🚀 Block-MLP [업그레이드! 2D 시계열 윈도우(T=8) + Temporal-SwiGLU 블록 신경망 + AdamW/Cosine]");
    println!("  [3] 1D-TCN (Dilated Residual Network)");
    println!("  [4] 1D-CNN (BatchNorm 적용 1차원 합성곱 신경망)");
    println!("  [5] LSTM (순환 신경망)");
    println!("  [6] CNN + LSTM (합성곱-순환 복합 모델)");
    println!("  [7] 전체 6종 모델 일괄 학습");
    println!("  💡 팁: 원하는 번호를 쉼표나 띄어쓰기로 입력하세요 (예: '2' 또는 '1,2' 또는 '2 3')");
    print!("선택 (기본 2: 업그레이드 Block-MLP) > ");
    io::stdout().flush()?;

    let mut model_input = String::new();
    io::stdin().read_line(&mut model_input)?;
    let trimmed = model_input.trim();

    let mut selected_archs = Vec::new();
    if trimmed.is_empty() || trimmed == "2" {
        selected_archs.push(ModelArchitecture::ModularMlp);
    } else if trimmed == "1" {
        selected_archs.push(ModelArchitecture::Mlp);
    } else if trimmed == "7" || trimmed.eq_ignore_ascii_case("all") {
        selected_archs = vec![
            ModelArchitecture::Mlp,
            ModelArchitecture::ModularMlp,
            ModelArchitecture::Tcn,
            ModelArchitecture::Cnn,
            ModelArchitecture::Lstm,
            ModelArchitecture::CnnLstm,
        ];
    } else {
        for token in trimmed.split(|c| c == ',' || c == ' ' || c == ';') {
            let t = token.trim();
            match t {
                "1" => {
                    if !selected_archs.contains(&ModelArchitecture::Mlp) {
                        selected_archs.push(ModelArchitecture::Mlp);
                    }
                }
                "2" => {
                    if !selected_archs.contains(&ModelArchitecture::ModularMlp) {
                        selected_archs.push(ModelArchitecture::ModularMlp);
                    }
                }
                "3" => {
                    if !selected_archs.contains(&ModelArchitecture::Tcn) {
                        selected_archs.push(ModelArchitecture::Tcn);
                    }
                }
                "4" => {
                    if !selected_archs.contains(&ModelArchitecture::Cnn) {
                        selected_archs.push(ModelArchitecture::Cnn);
                    }
                }
                "5" => {
                    if !selected_archs.contains(&ModelArchitecture::Lstm) {
                        selected_archs.push(ModelArchitecture::Lstm);
                    }
                }
                "6" => {
                    if !selected_archs.contains(&ModelArchitecture::CnnLstm) {
                        selected_archs.push(ModelArchitecture::CnnLstm);
                    }
                }
                _ => {}
            }
        }
    }

    if selected_archs.is_empty() {
        selected_archs.push(ModelArchitecture::ModularMlp);
    }

    let models_to_train = selected_archs;
    println!(
        "\n✔ [선택된 모델 목록 (총 {}개)]: {}",
        models_to_train.len(),
        models_to_train
            .iter()
            .map(|m| m.display_name())
            .collect::<Vec<_>>()
            .join(", ")
    );


    // 선택된 데이터셋의 고유 클래스 수 및 동작명 자동 감지
    let (detected_num_classes, detected_class_names) = {
        let mut max_l = 0usize;
        let mut names_map = std::collections::BTreeMap::new();
        if let Ok(file) = File::open(dataset_path) {
            let mut rdr = csv::Reader::from_reader(file);
            for rec in rdr.records().flatten() {
                if rec.len() >= 2 {
                    if let Ok(l) = rec[rec.len() - 2].parse::<usize>() {
                        max_l = max_l.max(l);
                        let name = rec[rec.len() - 1].trim().to_string();
                        names_map.entry(l).or_insert(name);
                    }
                }
            }
        }
        let nc = (max_l + 1).max(2);
        let mut names = Vec::new();
        for i in 0..nc {
            names.push(names_map.get(&i).cloned().unwrap_or_else(|| format!("Class {}", i)));
        }
        (nc, names)
    };

    println!(
        "\nℹ️ [데이터셋 클래스 자동 감지] 총 {}개 클래스 -> [{}]",
        detected_num_classes,
        detected_class_names.join(", ")
    );

    for arch in models_to_train {
        println!("\n========================================================");
        println!(" 🚀 {} 모델 GPU 학습 시작 (데이터셋: {})", arch.display_name(), dataset_path.display());
        println!("========================================================");

        let epochs = match arch {
            ModelArchitecture::Mlp => 12,
            ModelArchitecture::ModularMlp => 10,
            _ => 150,
        };

        let config = TrainingConfig {
            dataset_path: dataset_path.to_string_lossy().to_string(),
            model_arch: arch,
            epochs,
            learning_rate: 1e-3, // 안정화된 0.001 학습률
            batch_size: 64,
            num_classes: detected_num_classes,
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

/// 🚀 실시간 추론 시 제스처 안정화 엔진의 세부 판단 결과
#[derive(Clone, Debug)]
pub struct StabilizerDecision {
    pub stable_idx: usize,
    #[allow(dead_code)]
    pub stable_prob: f32,
    pub state_tag: &'static str,
    pub best_cand: usize,
    pub candidate_count: usize,
    pub required_debounce: usize,
    pub reason: String,
    pub is_event: bool,
}

/// 🚀 다중 파지(Multi-Grasp) 확장을 위한 대칭적 제스처 안정화 엔진 (Symmetric Multi-Pose Debouncer)
/// 1. 확률 지수이동평균(EMA): α = 0.35 (순수 신경망 Softmax 확률 벡터에 대해 편향 없이 스무딩)
/// 2. 생체역학적 수축 에너지 물리 인터락 (Physical Contraction Energy Gate):
///    - 능동 제스처(Class >= 2) 진입 시 물리적 실효 에너지(RMS >= (base_rest * 1.8).max(110.0)) 필수 검증
///    - 동적 잡음(Class 1) 상태에서의 일시적 모델 확률 튐에 의한 주먹 오탐 100% 원천 차단
/// 3. 상태 전이별 비대칭 디바운싱:
///    - 정적 휴식(0) -> 능동 제스처(>=2): 3프레임(약 60ms, p >= 50%) 초고속 진입
///    - 동적 잡음(1) -> 능동 제스처(>=2): 5프레임(약 100ms, p >= 60%) 엄격한 검증
///    - 능동 제스처(>=2) -> 휴식(0) 복귀: 5프레임(약 100ms) 자연스럽고 매끄러운 릴리즈
///    - 동적 잡음(1) -> 휴식(0) 복귀: 2프레임(약 40ms) 초고속 즉각 복귀
/// 4. 지능형 잡음 방어(Adaptive Noise Defense):
///    - 수축 유지 에너지(RMS >= 80.0) 존재 시 최대 10프레임(약 200ms) 동안 모터 진동/역기전력 방어
///    - 손을 펴서 에너지가 꺼졌거나 잡음이 200ms 이상 지속 시 즉시 동적 잡음(Class 1)으로 정상 탈출
/// 5. 모터 불응기(Refractory Period): 상태 전이 후 300ms 동안 역전이 차단
/// 6. 하드웨어 단선/이탈 침묵 보호: 실효 AC-RMS < 10.0 시 안전하게 Rest(0) 복귀
pub struct GestureStabilizer {
    ema_probs: Vec<f32>,
    current_stable_gesture: usize,
    candidate_gesture: usize,
    candidate_count: usize,
    last_transition_time: Instant,
    refractory_period: Duration,
    silence_rms: f32,
    base_rest_rms: f32,
    noise_hold_count: usize,
}

impl GestureStabilizer {
    pub fn new(num_classes: usize, base_rest_rms: f32) -> Self {
        let mut init_probs = vec![0.0f32; num_classes];
        if !init_probs.is_empty() {
            init_probs[0] = 1.0; // 초기 상태는 Rest
        }

        Self {
            ema_probs: init_probs,
            current_stable_gesture: 0,
            candidate_gesture: 0,
            candidate_count: 0,
            last_transition_time: Instant::now() - Duration::from_secs(1),
            refractory_period: Duration::from_millis(300),
            silence_rms: 10.0,
            base_rest_rms: base_rest_rms.max(25.0),
            noise_hold_count: 0,
        }
    }

    pub fn update(&mut self, raw_probs: &[f32], total_ac_rms: f32) -> StabilizerDecision {
        let num_classes = raw_probs.len();
        if self.ema_probs.len() != num_classes {
            self.ema_probs = vec![0.0; num_classes];
            self.ema_probs[0] = 1.0;
        }

        // 1. 순수 신경망 예측 확률 벡터 지수이동평균(EMA) 필터링 (α = 0.35: 빠른 반응성과 노이즈 감쇠의 최적 균형)
        let alpha = 0.35f32;
        for i in 0..num_classes {
            self.ema_probs[i] = alpha * raw_probs[i] + (1.0 - alpha) * self.ema_probs[i];
        }

        // 2. 센서 전극 분리 시 완전 침묵 안전 복귀
        if total_ac_rms < self.silence_rms {
            self.current_stable_gesture = 0;
            self.candidate_gesture = 0;
            self.candidate_count = 0;
            self.noise_hold_count = 0;
            let rest_p = self.ema_probs.get(0).copied().unwrap_or(1.0);
            return StabilizerDecision {
                stable_idx: 0,
                stable_prob: rest_p,
                state_tag: "침묵",
                best_cand: 0,
                candidate_count: 0,
                required_debounce: 1,
                reason: format!("센서 분리 침묵 감지 (RMS {:.1} < {:.1})", total_ac_rms, self.silence_rms),
                is_event: true,
            };
        }

        // 3. 모터 불응기(Refractory Period) 검사: 상태 전이 후 300ms 동안은 서보 회전 진동/노이즈에 의한 역전이 차단
        let elapsed_refractory = self.last_transition_time.elapsed();
        if elapsed_refractory < self.refractory_period {
            let p = self.ema_probs.get(self.current_stable_gesture).copied().unwrap_or(0.0);
            return StabilizerDecision {
                stable_idx: self.current_stable_gesture,
                stable_prob: p,
                state_tag: "잠금",
                best_cand: self.current_stable_gesture,
                candidate_count: 0,
                required_debounce: 0,
                reason: format!("모터 불응기 잠금 (경과 {}ms < 300ms)", elapsed_refractory.as_millis()),
                is_event: false,
            };
        }

        // 4. 최적 후보 탐색 (ArgMax over EMA Probabilities)
        let mut best_cand = 0;
        let mut best_p = self.ema_probs[0];
        for (i, &p) in self.ema_probs.iter().enumerate().skip(1) {
            if p > best_p {
                best_p = p;
                best_cand = i;
            }
        }

        // 🛡️ [생체역학적 수축 에너지 물리 인터락 (Physical Contraction Energy Gate)]
        // 동적 잡음(팔 흔들림, 옷깃 스침 등, RMS 60~100) 상황에서 신경망이 순간적으로 Class >= 2(주먹) 확률을 높게 주더라도
        // 물리적 수축 에너지가 주먹 최소 문턱치((base_rest * 1.8).max(110.0))에 미치지 못하면 주먹 진입을 100% 원천 차단합니다.
        let min_grasp_rms = (self.base_rest_rms * 1.8).max(110.0);
        if best_cand >= 2 && self.current_stable_gesture < 2 && total_ac_rms < min_grasp_rms {
            // 에너지가 부족한 상태에서의 주먹 확률은 동적 잡음 스파이크로 판정 -> Class 1(동적 잡음)으로 강제 리매핑
            best_cand = 1;
            best_p = self.ema_probs.get(1).copied().unwrap_or(0.5);
        }

        // 5. 다중 클래스 확장형 비대칭 디바운싱 및 히스테리시스 판정
        // - Class 0: 정적 휴식 (Rest)
        // - Class 1: 동적 잡음 / 일상 팔 움직임 (Noise / Non-Grasp)
        // - Class >= 2: 의도적 능동 제스처 (Active Grasp/Posture: 주먹, 손가락 펴기, 핀치 등)
        let is_currently_grasp = self.current_stable_gesture >= 2;
        let is_cand_noise = best_cand == 1; // Class 1: 동적 잡음

        // 상태 전이별 맞춤 디바운스 프레임 및 확률 임계치 (50Hz 기준, 1프레임 = 20ms)
        let (required_debounce, threshold) = if best_cand == 0 && is_currently_grasp {
            (5, 0.45f32) // 능동 제스처(Class >= 2)에서 휴식(0)으로 복귀: 5프레임(~100ms) 연속 검증 (자연스러운 릴리즈)
        } else if best_cand == 0 {
            (2, 0.45f32) // 동적 잡음(1)이나 기타 상태에서 휴식(0)으로 복귀: 2프레임(~40ms) 초고속 즉각 복귀!
        } else if best_cand == 1 && self.current_stable_gesture == 0 {
            (5, 0.50f32) // 정적 휴식(0)에서 동적 잡음(1)으로 튀는 것 방어: 5프레임(~100ms) 연속 검증
        } else if best_cand >= 2 && self.current_stable_gesture == 1 {
            // 🚀 [동적 잡음 중 주먹 진입 보호]: 잡음 상태는 신호가 불규칙하므로 5프레임(~100ms) 및 확실한 60% 확률 검증 요구!
            (5, 0.60f32)
        } else if best_cand >= 2 && self.current_stable_gesture >= 2 {
            (4, 0.50f32) // 제스처 간 상호 전환 (주먹 <-> 핀치): 4프레임(~80ms)
        } else {
            (3, 0.50f32) // 정적 휴식(0)에서 제스처(주먹 등) 진입: 직관적 초고속 3프레임(~60ms) 유지
        };

        let (state_tag, decision_reason, is_event);

        if is_currently_grasp && is_cand_noise {
            // 🛡️ [에너지 조건부 적응형 잡음 방어]
            // 주먹 유지에 필요한 최소 등척성 에너지 (휴식 기준선 대비 1.35배 또는 80.0 RMS 이상)
            let min_sustain_rms = (self.base_rest_rms * 1.35).max(80.0);

            if total_ac_rms >= min_sustain_rms && self.noise_hold_count < 10 {
                // 수축 에너지가 충분히 유지되고, 연속 잡음 방어 횟수가 10프레임(~200ms) 미만인 경우 악력 100% 유지
                self.candidate_count = 0;
                self.noise_hold_count += 1;
                state_tag = "유지";
                decision_reason = format!(
                    "수축 유지 중 잡음 방어 (Cand Class 1 무시 {}/10, RMS {:.1} >= {:.1}, p={:.1}%)",
                    self.noise_hold_count, total_ac_rms, min_sustain_rms, best_p * 100.0
                );
                is_event = false;
            } else {
                // 1) 사용자가 손을 펴서 RMS가 80.0 미만으로 떨어졌거나,
                // 2) 200ms(10프레임) 이상 지속적으로 Class 1(잡음)이 들어오는 경우 -> 주먹 수축이 완전히 해제된 상태!
                // 잡음을 무시하지 않고 정상적으로 Class 1(동적 잡음)으로 신속히 탈출 전이 진행
                if self.candidate_gesture == 1 {
                    self.candidate_count += 1;
                } else {
                    self.candidate_gesture = 1;
                    self.candidate_count = 1;
                }

                if self.candidate_count >= 3 {
                    let prev_idx = self.current_stable_gesture;
                    self.current_stable_gesture = 1;
                    self.candidate_count = 0;
                    self.noise_hold_count = 0;
                    self.last_transition_time = Instant::now();
                    state_tag = "확정";
                    decision_reason = format!(
                        "🎉 수축 해제 -> 동적 잡음 정상 복귀: Class {} -> Class 1 (RMS {:.1}, p={:.1}%)",
                        prev_idx, total_ac_rms, best_p * 100.0
                    );
                    is_event = true;
                } else {
                    state_tag = "전이";
                    decision_reason = format!(
                        "수축 해제 동적 잡음 전이 중: Class {} -> Class 1 ({}/3, RMS {:.1})",
                        self.current_stable_gesture, self.candidate_count, total_ac_rms
                    );
                    is_event = true;
                }
            }
        } else if best_cand == self.current_stable_gesture {
            self.candidate_gesture = best_cand;
            self.candidate_count = 0;
            self.noise_hold_count = 0;
            state_tag = "안정";
            decision_reason = format!("안정 유지 (p={:.1}%)", best_p * 100.0);
            is_event = false;
        } else if best_p >= threshold {
            self.noise_hold_count = 0;
            if best_cand == self.candidate_gesture && self.candidate_count > 0 {
                self.candidate_count += 1;
            } else {
                self.candidate_gesture = best_cand;
                self.candidate_count = 1;
            }

            if self.candidate_count >= required_debounce {
                let prev_idx = self.current_stable_gesture;
                self.current_stable_gesture = best_cand;
                self.candidate_count = 0;
                self.noise_hold_count = 0;
                self.last_transition_time = Instant::now();
                state_tag = "확정";
                decision_reason = format!("🎉 상태 전이 확정: Class {} -> Class {} (p={:.1}%)", prev_idx, best_cand, best_p * 100.0);
                is_event = true;
            } else {
                state_tag = "전이";
                decision_reason = format!("전이 진행: Class {} -> Class {} ({}/{}, p={:.1}%)", self.current_stable_gesture, best_cand, self.candidate_count, required_debounce, best_p * 100.0);
                is_event = true;
            }
        } else {
            self.candidate_count = 0;
            self.noise_hold_count = 0;
            state_tag = "유지";
            decision_reason = format!("전이 미달: Cand Class {} (p={:.1}% < {:.0}%)", best_cand, best_p * 100.0, threshold * 100.0);
            is_event = false;
        }

        let stable_p = self.ema_probs.get(self.current_stable_gesture).copied().unwrap_or(0.0);
        StabilizerDecision {
            stable_idx: self.current_stable_gesture,
            stable_prob: stable_p,
            state_tag,
            best_cand,
            candidate_count: self.candidate_count,
            required_debounce,
            reason: decision_reason,
            is_event,
        }
    }

    /// 각 클래스별 스무딩된 실시간 확률 벡터 반환
    pub fn get_probabilities(&self) -> &[f32] {
        &self.ema_probs
    }
}

// =========================================================================
// 3. 실시간 추론 (모델 선택 및 실시간 복합 추론)
// =========================================================================
fn run_inference_pipeline(device: &WgpuDevice) {
    println!("\n[추론에 사용할 모델 선택]");
    println!("  [1] MLP (emg_mlp_model.mpk) [1프레임 초고속 즉시추론]");
    println!("  [2] 🚀 Block-MLP (emg_modular_mlp_model.mpk) [🥇 업그레이드 추천! O(1) 링버퍼(T=8) + Temporal-SwiGLU 블록 신경망]");
    println!("  [3] 1D-TCN (emg_tcn_model.mpk)");
    println!("  [4] 1D-CNN (emg_cnn_model.mpk)");
    println!("  [5] LSTM (emg_lstm_model.mpk)");
    println!("  [6] CNN + LSTM (emg_cnnlstm_model.mpk)");
    println!("  [7] 4D LDA + SVM (emg_svm_model.json) [⚡ 4차원 판별 & 각 손가락 % 실시간 출력]");
    print!("선택 (1-7, 기본 2: 업그레이드 Block-MLP) > ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    if input.trim() == "7" {
        if let Err(e) = emg_ml_pipeline::lda::run_lda_svm_realtime_inference() {
            eprintln!("❌ 4D LDA & SVM 실시간 추론 에러: {}", e);
        }
        return;
    }

    let arch = match input.trim() {
        "1" => ModelArchitecture::Mlp,
        "2" | "" => ModelArchitecture::ModularMlp,
        "3" => ModelArchitecture::Tcn,
        "4" => ModelArchitecture::Cnn,
        "5" => ModelArchitecture::Lstm,
        "6" => ModelArchitecture::CnnLstm,
        _ => ModelArchitecture::ModularMlp,
    };


    let model_file_name = arch.file_name();
    let model_file_path = format!("{}.mpk", model_file_name);
    let fallback_mlp_path = "model_artifacts/final_model.mpk";
    if !Path::new(&model_file_path).exists() && !(arch == ModelArchitecture::Mlp && Path::new(fallback_mlp_path).exists()) {
        println!(
            "\n❌ [에러] '{}' 모델 파일이 없습니다. 먼저 2번 로컬 학습을 진행해주세요.",
            model_file_path
        );
        return;
    }

    // 학습 시 저장된 Z-Score 정규화 통계 로드 및 클래스 수 결정
    let norm_stats = NormalizationStats::load_for_model(model_file_name);
    let feature_dim = norm_stats.as_ref().map(|s| s.feature_dim).unwrap_or(TOTAL_FEATURES_ALL);
    let lda_model_opt: Option<emg_ml_pipeline::LdaModel> = if feature_dim == 4 {
        match File::open("emg_lda4d_model.json").ok().and_then(|f| serde_json::from_reader(f).ok()) {
            Some(lda) => {
                println!("✔ [4D LDA 하이브리드 파이프라인 활성화] 5ch 센서 ➔ 75D FFT ➔ 4D LDA 투영 ➔ {} 딥러닝 추론!", arch.display_name());
                Some(lda)
            }
            None => {
                eprintln!("❌ [에러] 4D LDA 모델 파일('emg_lda4d_model.json')을 로드할 수 없습니다.");
                eprintln!("   먼저 메뉴 [5] 또는 4D LDA 생성을 실행해주세요.");
                return;
            }
        }
    } else {
        None
    };
    let num_classes = norm_stats.as_ref().map(|s| s.num_classes).unwrap_or(NUM_CLASSES).max(2);

    let action_names: Vec<String> = if let Some(stats) = &norm_stats {
        if !stats.class_names.is_empty() {
            stats.class_names.clone()
        } else if num_classes == 2 {
            vec![
                "휴식 (Relax)".into(),
                "손 쥐기 (Fist)".into(),
            ]
        } else if num_classes == 3 {
            vec![
                "정적 휴식 (Rest)".into(),
                "동적 잡음 (Noise)".into(),
                "진짜 주먹 (Fist)".into(),
            ]
        } else if num_classes == 6 {
            vec![
                "휴식 (Relax)".into(),
                "엄지 (Thumb)".into(),
                "검지 (Index)".into(),
                "중지 (Middle)".into(),
                "약지 (Ring)".into(),
                "새끼 (Pinky)".into(),
            ]
        } else {
            vec![
                "휴식 (Relax)".into(),
                "주먹 쥐기 (Fist)".into(),
                "손가락 펴기 (Open)".into(),
                "엄지 굽히기 (Thumb)".into(),
            ]
        }
    } else if num_classes == 2 {
        vec![
            "휴식 (Relax)".into(),
            "손 쥐기 (Fist)".into(),
        ]
    } else if num_classes == 3 {
        vec![
            "정적 휴식 (Rest)".into(),
            "동적 잡음 (Noise)".into(),
            "진짜 주먹 (Fist)".into(),
        ]
    } else if num_classes == 6 {
        vec![
            "휴식 (Relax)".into(),
            "엄지 (Thumb)".into(),
            "검지 (Index)".into(),
            "중지 (Middle)".into(),
            "약지 (Ring)".into(),
            "새끼 (Pinky)".into(),
        ]
    } else {
        vec![
            "휴식 (Relax)".into(),
            "주먹 쥐기 (Fist)".into(),
            "손가락 펴기 (Open)".into(),
            "엄지 굽히기 (Thumb)".into(),
        ]
    };

    if norm_stats.is_some() {
        println!(
            "✔ [정규화 통계 로드 완료] 학습 당시 Z-Score 표준화 적용 (클래스: {}개 - {})",
            num_classes,
            action_names.join(", ")
        );
    } else {
        println!("ℹ️ 정규화 통계 파일 없음 (기본 스케일링 모드로 작동)");
    }

    println!("\n[알림] 저장된 {} 모델 로드 중 (특징 차원: {}, 클래스: {}개)...", arch.display_name(), feature_dim, num_classes);
    let recorder = CompactRecorder::new();

    enum LoadedModel {
        Mlp(EmgMlpModel<MyBackend>),
        ModularMlp(EmgModularMlpModel<MyBackend>),
        Tcn(EmgTcnModel<MyBackend>),
        Cnn(EmgCnnModel<MyBackend>),
        Lstm(EmgLstmModel<MyBackend>),
        CnnLstm(EmgCnnLstmModel<MyBackend>),
    }

    let loaded_model = match arch {
        ModelArchitecture::Mlp => {
            let mut m = EmgMlpModel::new(device, feature_dim, num_classes);
            let load_target = if Path::new(&model_file_path).exists() {
                model_file_name
            } else if Path::new(fallback_mlp_path).exists() {
                "model_artifacts/final_model"
            } else {
                model_file_name
            };
            match recorder.load(load_target.into(), device) {
                Ok(rec) => m = m.load_record(rec),
                Err(e) => {
                    eprintln!("❌ MLP 가중치 로드 실패: {:?}", e);
                    return;
                }
            }
            LoadedModel::Mlp(m)
        }
        ModelArchitecture::ModularMlp => {
            let mut m = EmgModularMlpModel::new(device, feature_dim, num_classes);
            match recorder.load(model_file_name.into(), device) {
                Ok(rec) => m = m.load_record(rec),
                Err(e) => {
                    eprintln!("❌ Block-MLP 가중치 로드 실패: {:?}", e);
                    return;
                }
            }
            LoadedModel::ModularMlp(m)
        }
        ModelArchitecture::Tcn => {
            let mut m = EmgTcnModel::new(device, feature_dim, num_classes);
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
            let mut m = EmgCnnModel::new(device, feature_dim, num_classes);
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
            let mut m = EmgLstmModel::new(device, feature_dim, num_classes);
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
            let mut m = EmgCnnLstmModel::new(device, feature_dim, num_classes);
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

    let mut port_write = port.try_clone().ok();
    if port_write.is_some() {
        println!("✔ 아두이노 양방향 시리얼 통신 활성화 (0:정적휴식/잡음, 1:진짜주먹 역전송)");
    } else {
        println!("⚠️ 시리얼 포트 복제 실패: 단방향(수신 전용)으로 실행됩니다.");
    }

    let mut reader = BufReader::new(port);

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
    let mut calibrated_rest_rms = 45.0f32; // 기본 휴식 기준선

    if !baseline_samples.is_empty() {
        for s in &baseline_samples {
            for ch in 0..NUM_CHANNELS {
                rest_baseline[ch] += s[ch];
            }
        }
        for ch in 0..NUM_CHANNELS {
            rest_baseline[ch] /= baseline_samples.len() as f32;
        }

        // 실시간 휴식기 총 AC-RMS 동적 산출 (안정화 게이트 기준선)
        let mut cal_proc = EmgFftProcessor::new();
        let mut rms_sum = 0.0f32;
        let mut valid_cal_count = 0;
        for (idx, s) in baseline_samples.iter().enumerate() {
            let feats75 = cal_proc.process_sample_all(s);
            if idx >= 16 { // FFT 윈도우(16) 충전 이후의 신뢰도 높은 RMS만 합산
                rms_sum += EmgFftProcessor::extract_total_ac_rms(&feats75);
                valid_cal_count += 1;
            }
        }
        if valid_cal_count > 0 {
            calibrated_rest_rms = (rms_sum / valid_cal_count as f32).max(10.0);
        }

        println!(
            "\n✔ [영점 보정 완료] 현재 착용 기준선: [Ch0: {:.0}, Ch1: {:.0}, Ch2: {:.0}, Ch3: {:.0}, Ch4: {:.0}] | 실측 휴식 RMS: {:.1}",
            rest_baseline[0], rest_baseline[1], rest_baseline[2], rest_baseline[3], rest_baseline[4], calibrated_rest_rms
        );
    } else {
        println!("\nℹ️ [영점 보정 건너뜀] 즉시 추론을 시작합니다. (기본 휴식 RMS: {:.1})", calibrated_rest_rms);
    }

    // 🚀 [실시간 세션 간 진폭 스케일 자동 정렬 (Live Baseline Gain Calibration)]
    // 학습 데이터셋의 휴식기 기대 RMS(z ≈ -0.60 지점)와 현재 착용 시 실측된 휴식기 RMS(calibrated_rest_rms)의
    // 비율(cal_gain)을 산출하여, 전극 밀착도/시간 경과에 따른 기준선 드리프트를 실시간으로 자동 보정합니다.
    // 🚀 [신호 정합 복원] 임의 게인 스케일링으로 인한 진폭 왜곡(0.4x 압살)을 완전히 폐기하고
    // 센서에서 추출된 원본 피처를 Z-Score 표준화에 100% 온전히 전달합니다.
    let cal_gain = 1.0f32;

    /// 진폭 비례 피처(Raw AC, FFT Mag, 시간도메인의 RMS, MAV, WL)에 실시간 영점 보정 게인을 적용
    #[inline]
    fn apply_calibration_gain_slice(feats: &mut [f32], gain: f32) {
        if (gain - 1.0).abs() < 1e-3 {
            return;
        }
        if feats.len() >= TOTAL_FEATURES_ALL {
            // 75D: 0..45(Raw + FFT) 스케일링
            for v in feats[0..45].iter_mut() {
                *v *= gain;
            }
            // 45..75: 각 채널별 ac_rms, ac_mav, wl 스케일링 (zc, ssc, wamp는 주파수 불변 유지)
            for ch in 0..NUM_CHANNELS {
                let base = 45 + ch * 6;
                feats[base] *= gain;     // ac_rms
                feats[base + 1] *= gain; // ac_mav
                feats[base + 2] *= gain; // wl
            }
        } else if feats.len() >= TD_CORE_FEATURES {
            // 35D: 0..5(Raw 평활) 스케일링
            for v in feats[0..NUM_CHANNELS].iter_mut() {
                *v *= gain;
            }
            // 5..35: 각 채널별 ac_rms, ac_mav, wl 스케일링
            for ch in 0..NUM_CHANNELS {
                let base = NUM_CHANNELS + ch * 6;
                feats[base] *= gain;     // ac_rms
                feats[base + 1] *= gain; // ac_mav
                feats[base + 2] *= gain; // wl
            }
        }
    }

    // 🚀 [비동기 시리얼 고속 수신 스레드]
    // 메인 추론 스레드를 시리얼 I/O 블로킹(타임아웃, 커널 락)으로부터 100% 분리 격리합니다.
    let (data_tx, data_rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(150);
    let _reader_thread = std::thread::spawn(move || {
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let parts: Vec<f32> = trimmed
                        .split(',')
                        .filter_map(|s| s.trim().parse::<f32>().ok())
                        .collect();
                    if parts.len() == NUM_CHANNELS || parts.len() >= 75 {
                        let _ = data_tx.try_send(parts); // 큐가 꽉 차면 지연 방지를 위해 즉시 드롭
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::TimedOut => continue,
                Err(_) => break,
            }
        }
    });

    let mut processor = EmgFftProcessor::new();
    let mut window_buffer: VecDeque<Vec<f32>> = VecDeque::with_capacity(SEQ_LEN);
    // 🚀 [단계 1] Block-MLP 전용 O(1) 순환 링 버퍼 (FIFO Circular Queue + 첫 프레임 복제 패딩)
    let mut modular_ring_buffer =
        RealTimeRingBuffer::new(MODULAR_MLP_WINDOW_SIZE, feature_dim, None, None);
    // 🚀 [지능형 제스처 안정화 엔진] 확률 EMA + 모터 불응기(350ms) + 비대칭 히스테리시스
    let mut stabilizer = GestureStabilizer::new(num_classes, calibrated_rest_rms);
    let mut last_sent_val: Option<u8> = None;
    let mut frame_count: u64 = 0;

    // 📝 [상세 디버그 로그 파일 생성]
    let log_file_path = "inference_debug.log";
    let mut log_writer = std::fs::File::create(log_file_path)
        .map(|f| std::io::BufWriter::new(f))
        .ok();
    if let Some(ref mut writer) = log_writer {
        use std::io::Write;
        let _ = writeln!(
            writer,
            "=== EMG 실시간 추론 상세 디버그 로그 시작 ({}) | 아키텍처: {} | 기준선 RMS: {:.1} | Gain: x{:.2} ===",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            arch.display_name(),
            calibrated_rest_rms,
            cal_gain
        );
        let _ = writeln!(
            writer,
            "TIME\tFRAME\tSTATE\tACTION\tCANDIDATE\tCOUNT\tRMS\tRMS_RATIO\tRAW_PROBS\tEMA_PROBS\tSENSORS\tTX\tREASON"
        );
        let _ = writer.flush();
    }
    let inference_start_time = std::time::Instant::now();

    println!("\n==========================================================");
    println!(" 🚀 {} 실시간 추론 시작! (비동기 수신 & 350ms 모터 불응기 적용 / 종료: Ctrl + C)", arch.display_name());
    println!(" 📝 [디버그 로그] 세부 판정 로그가 '{}'에 실시간 기록됩니다.", log_file_path);
    println!("==========================================================");

    loop {
        // 첫 번째 샘플을 최대 50ms 대기 (채널이 닫히면 아두이노 연결 해제 감지)
        let first_sample = match data_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(s) => s,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                eprintln!("\n⚠️ 시리얼 포트 연결이 끊어졌거나 아두이노가 재부팅되었습니다. 실시간 추론을 종료합니다.");
                break;
            }
        };

        // 🚀 채널에 쌓여있는 모든 중간 프레임을 고속 인출 (Zero-Latency 드레인)
        let mut batch_samples = vec![first_sample];
        while let Ok(extra) = data_rx.try_recv() {
            batch_samples.push(extra);
            if batch_samples.len() >= 30 {
                break;
            }
        }

        let mut latest_features: Option<Vec<f32>> = None;
        let mut latest_ac_rms = 0.0f32;
        let mut latest_raw = [0.0f32; NUM_CHANNELS];

        for parts in &batch_samples {
            if parts.len() >= 75 {
                let (cur, rms) = if feature_dim == TD_CORE_FEATURES {
                    let mut feats35 = [0.0f32; TD_CORE_FEATURES];
                    feats35[..NUM_CHANNELS].copy_from_slice(&parts[..NUM_CHANNELS]);
                    feats35[NUM_CHANNELS..].copy_from_slice(&parts[45..75]);
                    let r = EmgFftProcessor::extract_total_ac_rms_td35(&feats35);
                    apply_calibration_gain_slice(&mut feats35, cal_gain);
                    (feats35.to_vec(), r)
                } else {
                    let feats75 = &parts[..75];
                    let mut feats_arr: [f32; TOTAL_FEATURES_ALL] = match feats75.try_into() {
                        Ok(arr) => arr,
                        Err(_) => continue,
                    };
                    let r = EmgFftProcessor::extract_total_ac_rms(&feats_arr);
                    apply_calibration_gain_slice(&mut feats_arr, cal_gain);
                    let c = if let Some(ref lda) = lda_model_opt {
                        lda.transform_single(&feats_arr)
                    } else {
                        feats_arr.to_vec()
                    };
                    (c, r)
                };
                let norm = if let Some(ref stats) = norm_stats {
                    stats.normalize(&cur)
                } else {
                    cur.clone()
                };

                modular_ring_buffer.push_normalized_and_get_window(&norm);
                if window_buffer.len() == SEQ_LEN {
                    window_buffer.pop_front();
                }
                window_buffer.push_back(norm);

                latest_features = Some(cur);
                latest_ac_rms = rms;
            } else if parts.len() == NUM_CHANNELS {
                let mut raw_arr = [0.0f32; NUM_CHANNELS];
                raw_arr.copy_from_slice(&parts[..NUM_CHANNELS]);
                latest_raw = raw_arr;

                let (cur, rms) = if feature_dim == TD_CORE_FEATURES {
                    // 🚀 [35D 경량 시간도메인 모드] FFT(40차원)를 완전히 생략하여 초저지연 연산!
                    let mut feats35 = processor.process_sample_td35(&raw_arr);
                    let r = EmgFftProcessor::extract_total_ac_rms_td35(&feats35);
                    apply_calibration_gain_slice(&mut feats35, cal_gain);
                    (feats35.to_vec(), r)
                } else if feature_dim == TOTAL_FEATURES_ALL {
                    let mut feats75 = processor.process_sample_all(&raw_arr);
                    let r = EmgFftProcessor::extract_total_ac_rms(&feats75);
                    apply_calibration_gain_slice(&mut feats75, cal_gain);
                    let c = if let Some(ref lda) = lda_model_opt {
                        lda.transform_single(&feats75)
                    } else {
                        feats75.to_vec()
                    };
                    (c, r)
                } else if feature_dim == TOTAL_FEATURES_WITH_RAW {
                    let feats = processor.process_sample_combined(&raw_arr);
                    (feats.to_vec(), raw_arr.iter().map(|x| x.abs()).sum::<f32>() / 5.0)
                } else {
                    (raw_arr.to_vec(), raw_arr.iter().map(|x| x.abs()).sum::<f32>() / 5.0)
                };

                let norm = if let Some(ref stats) = norm_stats {
                    stats.normalize(&cur)
                } else {
                    cur.clone()
                };

                modular_ring_buffer.push_normalized_and_get_window(&norm);
                if window_buffer.len() == SEQ_LEN {
                    window_buffer.pop_front();
                }
                window_buffer.push_back(norm);

                latest_features = Some(cur);
                latest_ac_rms = rms;
            }
        }

        let current_features = match latest_features {
            Some(f) => f,
            None => continue,
        };
        let total_ac_rms = latest_ac_rms;

        let normalized: Vec<f32> = if let Some(ref stats) = norm_stats {
            stats.normalize(&current_features)
        } else {
            current_features.clone()
        };

        let logits = match &loaded_model {
            LoadedModel::Mlp(m) => {
                let input_2d = Tensor::<MyBackend, 1>::from_floats(
                    normalized.as_slice(),
                    device,
                )
                .reshape([1, feature_dim]);
                m.forward(input_2d)
            }
            LoadedModel::ModularMlp(m) => {
                let input_3d = modular_ring_buffer.get_tensor::<MyBackend>(device);
                m.forward(input_3d)
            }
            LoadedModel::Tcn(m) => {
                let mut seq_data = Vec::with_capacity(SEQ_LEN * feature_dim);
                let first_elem = window_buffer.front().cloned().unwrap_or_else(|| normalized.clone());
                for _ in 0..(SEQ_LEN.saturating_sub(window_buffer.len())) {
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
                m.forward(input_tensor)
            }
            LoadedModel::Cnn(m) => {
                let mut seq_data = Vec::with_capacity(SEQ_LEN * feature_dim);
                let first_elem = window_buffer.front().cloned().unwrap_or_else(|| normalized.clone());
                for _ in 0..(SEQ_LEN.saturating_sub(window_buffer.len())) {
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
                m.forward(input_tensor)
            }
            LoadedModel::Lstm(m) => {
                let mut seq_data = Vec::with_capacity(SEQ_LEN * feature_dim);
                let first_elem = window_buffer.front().cloned().unwrap_or_else(|| normalized.clone());
                for _ in 0..(SEQ_LEN.saturating_sub(window_buffer.len())) {
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
                m.forward(input_tensor)
            }
            LoadedModel::CnnLstm(m) => {
                let mut seq_data = Vec::with_capacity(SEQ_LEN * feature_dim);
                let first_elem = window_buffer.front().cloned().unwrap_or_else(|| normalized.clone());
                for _ in 0..(SEQ_LEN.saturating_sub(window_buffer.len())) {
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
                m.forward(input_tensor)
            }
        };

        let logits_vec = logits.into_data().to_vec::<f32>().unwrap();

        // 🚀 Softmax 확률 계산 (표준 Softmax T = 1.0 적용으로 클래스 간 변별력 보존)
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

        // 🚀 지능형 제스처 안정화 (확률 EMA + 350ms 불응기 + 비대칭 히스테리시스 + 상세 진단 로그)
        let decision = stabilizer.update(&raw_probs, total_ac_rms);
        let stable_idx = decision.stable_idx;
        let state_tag = decision.state_tag;
        let probs = stabilizer.get_probabilities();
        let prob_str = action_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let p = probs.get(i).copied().unwrap_or(0.0) * 100.0;
                let short_name = name.split_whitespace().next().unwrap_or(name);
                format!("{}:{:4.1}%", short_name, p)
            })
            .collect::<Vec<_>>()
            .join(" | ");

        let action_name = action_names.get(stable_idx).map(|s| s.as_str()).unwrap_or("알 수 없음");
        let cand_name = action_names.get(decision.best_cand).map(|s| s.as_str()).unwrap_or("알 수 없음");
        // 🚀 [아두이노 역전송 판별] 진짜 주먹 / 손 쥐기인 경우 1, 그 외(휴식, 동적 잡음)는 즉시 0 (잠금 및 래치 방지)
        let is_fist = action_name.contains("주먹") || action_name.contains("손 쥐기") || action_name.to_lowercase().contains("fist");
        let out_val: u8 = if is_fist { 1 } else { 0 };

        // 상태 변경 시 즉시 전송 + 25프레임(~0.5초) 주기적 동기화 하트비트 전송
        frame_count = frame_count.wrapping_add(1);
        let should_send = match last_sent_val {
            Some(last) => last != out_val || frame_count % 25 == 0,
            None => true,
        };

        let mut tx_occurred = false;
        if should_send {
            if let Some(ref mut writer) = port_write {
                let msg = format!("{}\n", out_val);
                if writer.write_all(msg.as_bytes()).is_ok() {
                    // ⚠️ writer.flush() 호출 제거: tcdrain() 동기화 블로킹으로 인한 실시간 추론 프리징 원천 방지
                    last_sent_val = Some(out_val);
                    tx_occurred = true;
                }
            }
        }

        // 📝 파일 디버그 로그 기록 (inference_debug.log)
        if let Some(ref mut writer) = log_writer {
            use std::io::Write;
            let raw_str = raw_probs.iter().map(|p| format!("{:.1}%", p * 100.0)).collect::<Vec<_>>().join("/");
            let ema_str = probs.iter().map(|p| format!("{:.1}%", p * 100.0)).collect::<Vec<_>>().join("/");
            let rms_ratio = total_ac_rms / calibrated_rest_rms.max(1.0);
            let _ = writeln!(
                writer,
                "{:>7.3}\t{}\t{}\t{}\t{}\t{}/{}\t{:.1}\tx{:.2}\t[{}]\t[{}]\t[{:.0},{:.0},{:.0},{:.0},{:.0}]\t{}\t{}",
                inference_start_time.elapsed().as_secs_f32(),
                frame_count,
                decision.state_tag,
                action_name,
                cand_name,
                decision.candidate_count,
                decision.required_debounce,
                total_ac_rms,
                rms_ratio,
                raw_str,
                ema_str,
                latest_raw[0], latest_raw[1], latest_raw[2], latest_raw[3], latest_raw[4],
                if tx_occurred { out_val.to_string() } else { "-".to_string() },
                decision.reason
            );
            if decision.is_event || frame_count % 25 == 0 {
                let _ = writer.flush();
            }
        }

        // ⚡ 콘솔 이벤트 개행 영구 출력 (주먹 진입 시도, 유지 락 발동, 상태 확정, 탈출 등 유의미한 동작 시 줄바꿈 출력)
        if decision.is_event {
            let rms_ratio = total_ac_rms / calibrated_rest_rms.max(1.0);
            println!(
                "\n⚡ [{:>6.2}s|F{:05}] [{:>2}] 현재:{:<8} -> 후보:{:<8} ({}/{}) | RMS:{:4.1}(x{:.2}) | {}",
                inference_start_time.elapsed().as_secs_f32(),
                frame_count,
                decision.state_tag,
                action_name,
                cand_name,
                decision.candidate_count,
                decision.required_debounce,
                total_ac_rms,
                rms_ratio,
                decision.reason
            );
        }

        let tx_tag = if out_val == 1 { "📤아두이노: 1 (주먹)" } else { "📤아두이노: 0 (휴식)" };

        // 센서 전극 들뜸(포화) 감지
        let mut sat_ch: Option<usize> = None;
        for ch in 0..NUM_CHANNELS {
            if latest_raw[ch] >= 1020.0 {
                sat_ch = Some(ch);
                break;
            }
        }
        let sat_str = if let Some(ch) = sat_ch {
            format!(" ⚠️Ch{}포화!", ch)
        } else {
            "".to_string()
        };

        let rms_ratio = total_ac_rms / calibrated_rest_rms.max(1.0);
        print!(
            "\r🤖 [{:<8}] [{:>2}] {:<12} | [{}] | RMS:{:4.1}(x{:.2}){} | {}  ",
            arch.display_name(),
            state_tag,
            action_name,
            prob_str,
            total_ac_rms,
            rms_ratio,
            sat_str,
            tx_tag,
        );
        io::stdout().flush().unwrap();
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

    found_files.sort();
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
    println!("  [1] 🚀 Block-MLP (🥇 업그레이드 추천! Temporal-SwiGLU 블록 신경망) (`modular_mlp`)");
    println!("  [2] 1D-TCN (Temporal Convolutional Network) (`tcn`)");
    println!("  [3] 1D-CNN (`cnn`)");
    println!("  [4] LSTM (`lstm`)");
    println!("  [5] CNN + LSTM (`cnn_lstm`)");
    print!("선택 (1-5, 기본 1) > ");
    io::stdout().flush()?;

    let mut model_choice = String::new();
    io::stdin().read_line(&mut model_choice)?;
    let model_type_str = match model_choice.trim() {
        "2" => "tcn",
        "3" => "cnn",
        "4" => "lstm",
        "5" => "cnn_lstm",
        _ => "modular_mlp",
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
            "modular_mlp" => "emg_modular_mlp_model.mpk".to_string(),
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

// =========================================================================
// 4. 정적 데이터셋 검증 테스트 (Inference Test - 무작위 샘플 고정밀 검증)
// =========================================================================
fn run_static_dataset_test(device: &WgpuDevice) {
    let mut found_files = Vec::new();
    for entry in glob("emg_dataset_*.csv").expect("Failed glob") {
        if let Ok(path) = entry {
            if std::fs::metadata(&path).map(|m| m.len() > 600).unwrap_or(false) {
                found_files.push(path);
            }
        }
    }
    for entry in glob("mlp/*.csv").expect("Failed glob") {
        if let Ok(path) = entry {
            if std::fs::metadata(&path).map(|m| m.len() > 600).unwrap_or(false) {
                found_files.push(path);
            }
        }
    }
    for entry in glob("uploads/*.csv").expect("Failed glob") {
        if let Ok(path) = entry {
            if std::fs::metadata(&path).map(|m| m.len() > 600).unwrap_or(false) {
                found_files.push(path);
            }
        }
    }

    found_files.sort();
    if found_files.is_empty() {
        println!("\n❌ [에러] CSV 데이터셋 파일이 없습니다. 먼저 1번 데이터 수집을 진행해주세요.");
        return;
    }

    println!("\n[검증 테스트에 사용할 데이터셋 파일 선택]");
    for (i, file) in found_files.iter().enumerate() {
        println!("  [{}] {}", i + 1, file.display());
    }
    print!("선택 (기본 1) > ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let choice = input.trim().parse::<usize>().unwrap_or(1);
    let dataset_path = if choice > 0 && choice <= found_files.len() {
        &found_files[choice - 1]
    } else {
        &found_files[0]
    };

    println!("\n[검증할 MLP 아키텍처 선택]");
    println!("  [1] 기본 1D MLP (emg_mlp_model.mpk)");
    println!("  [2] 🚀 업그레이드 Block-MLP (emg_modular_mlp_model.mpk - 2D 윈도우 + Temporal-SwiGLU 블록 신경망)");
    print!("선택 (1-2, 기본 2: 업그레이드 Block-MLP) > ");
    io::stdout().flush().unwrap();
    let mut arch_in = String::new();
    io::stdin().read_line(&mut arch_in).unwrap();
    let use_modular = arch_in.trim() != "1";

    let model_file_name = if use_modular {
        "emg_modular_mlp_model"
    } else {
        "emg_mlp_model"
    };
    let model_file_path = format!("{}.mpk", model_file_name);
    let fallback_path = "model_artifacts/final_model.mpk";

    let load_target = if Path::new(&model_file_path).exists() {
        model_file_name
    } else if !use_modular && Path::new(fallback_path).exists() {
        "model_artifacts/final_model"
    } else {
        println!(
            "\n❌ [에러] '{}' 파일이 없습니다. 먼저 2번 로컬 학습을 진행해주세요.",
            model_file_path
        );
        return;
    };

    let stats = match NormalizationStats::load_for_model(model_file_name) {
        Some(s) => s,
        None => {
            if let Ok(f) = File::open("model_artifacts/norm_stats.json") {
                serde_json::from_reader(f).unwrap_or(NormalizationStats {
                    feature_dim: 75,
                    means: vec![0.0; 75],
                    stds: vec![1.0; 75],
                    num_classes: NUM_CLASSES,
                    class_names: vec![],
                })
            } else {
                println!("❌ 정규화 통계 파일을 찾을 수 없습니다. 먼저 학습을 진행해주세요.");
                return;
            }
        }
    };

    let num_classes = stats.num_classes.max(2);
    let action_names: Vec<String> = if !stats.class_names.is_empty() {
        stats.class_names.clone()
    } else if num_classes == 2 {
        vec![
            "휴식 (Relax)".into(),
            "손 쥐기 (Fist)".into(),
        ]
    } else if num_classes == 3 {
        vec![
            "정적 휴식 (Rest)".into(),
            "동적 잡음 (Noise)".into(),
            "진짜 주먹 (Fist)".into(),
        ]
    } else if num_classes == 6 {
        vec![
            "휴식 (Relax)".into(),
            "엄지 (Thumb)".into(),
            "검지 (Index)".into(),
            "중지 (Middle)".into(),
            "약지 (Ring)".into(),
            "새끼 (Pinky)".into(),
        ]
    } else {
        vec![
            "휴식 (Relax)".into(),
            "주먹 쥐기 (Fist)".into(),
            "손가락 펴기 (Open)".into(),
            "엄지 굽히기 (Thumb Flex)".into(),
        ]
    };

    let recorder = CompactRecorder::new();

    if use_modular {
        let mut model = EmgModularMlpModel::new(device, stats.feature_dim, num_classes);

        match recorder.load(load_target.into(), device) {
            Ok(rec) => model = model.load_record(rec),
            Err(e) => {
                eprintln!("❌ Block-MLP 가중치 로드 실패: {:?}", e);
                return;
            }
        }

        println!(
            "\n✔ Block-MLP (Temporal-SwiGLU) 모델 로드 완료! (가중치: {}, 윈도우: T={}, 클래스: {}개)",
            load_target, MODULAR_MLP_WINDOW_SIZE, num_classes
        );

        let (_, valid_dataset, _) = match load_modular_mlp_dataset(
            &dataset_path.to_string_lossy(),
            MODULAR_MLP_WINDOW_SIZE,
        ) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("❌ 시계열 윈도우 데이터셋 로드 실패: {:?}", e);
                return;
            }
        };

        println!("\n=== 🎯 Block-MLP 정적 검증 데이터셋 무작위 5회 추론 테스트 ===");
        let mut rng = rand::thread_rng();
        let sample_choices: Vec<_> = valid_dataset
            .items
            .choose_multiple(&mut rng, 5)
            .cloned()
            .collect();

        for (idx, sample) in sample_choices.iter().enumerate() {
            let start_time = Instant::now();
            let input_tensor =
                Tensor::<MyBackend, 1>::from_floats(sample.window.as_slice(), device).reshape([
                    1,
                    sample.seq_len,
                    sample.feature_dim,
                ]);

            let logits = model.forward(input_tensor);
            let probs = burn::tensor::activation::softmax(logits.clone(), 1);
            let pred_label: i32 = logits.argmax(1).into_scalar();
            let latency = start_time.elapsed();
            let probs_vec: Vec<f32> = probs.into_data().to_vec().unwrap();

            let mark = if sample.label == pred_label as usize {
                "✔ 일치"
            } else {
                "❌ 불일치"
            };
            println!(
                "샘플 #{}: 정답: [{}] -> 예측: [{}] {} (소요시간: {:?})",
                idx + 1,
                action_names
                    .get(sample.label)
                    .map(|s| s.as_str())
                    .unwrap_or("알 수 없음"),
                action_names
                    .get(pred_label as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("알 수 없음"),
                mark,
                latency
            );
            let prob_line = action_names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let p = probs_vec.get(i).copied().unwrap_or(0.0) * 100.0;
                    let short_name = name.split_whitespace().next().unwrap_or(name);
                    format!("{}: {:.1}%", short_name, p)
                })
                .collect::<Vec<_>>()
                .join(", ");
            println!("   확률: {}", prob_line);
        }
        return;
    }

    let mut model = EmgMlpConfig {
        d_input: stats.feature_dim,
        d_hidden1: 128,
        d_hidden2: 64,
        d_hidden3: 32,
        d_hidden4: 16,
        num_classes,
        dropout: 0.15,
    }
    .init::<MyBackend>(device);

    match recorder.load(load_target.into(), device) {
        Ok(rec) => model = model.load_record(rec),
        Err(e) => {
            eprintln!("❌ MLP 가중치 로드 실패: {:?}", e);
            return;
        }
    }

    println!("\n✔ MLP 모델 로드 완료! (가중치: {}, 클래스: {}개)", load_target, num_classes);

    let (_, valid_dataset, _) = match load_mlp_dataset(&dataset_path.to_string_lossy()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("❌ 데이터셋 로드 실패: {:?}", e);
            return;
        }
    };

    println!("\n=== 🎯 정적 검증 데이터셋 무작위 5회 추론 테스트 ===");
    let mut rng = rand::thread_rng();
    let sample_choices: Vec<_> = valid_dataset.items.choose_multiple(&mut rng, 5).cloned().collect();

    for (idx, sample) in sample_choices.iter().enumerate() {
        let start_time = Instant::now();
        let input_tensor = Tensor::<MyBackend, 1>::from_floats(sample.features.as_slice(), device)
            .reshape([1, stats.feature_dim]);

        let logits = model.forward(input_tensor);
        let probs = burn::tensor::activation::softmax(logits.clone(), 1);
        let pred_label: i32 = logits.argmax(1).into_scalar();
        let latency = start_time.elapsed();
        let probs_vec: Vec<f32> = probs.into_data().to_vec().unwrap();

        let mark = if sample.label == pred_label as usize { "✔ 일치" } else { "❌ 불일치" };
        println!(
            "샘플 #{}: 정답: [{}] -> 예측: [{}] {} (소요시간: {:?})",
            idx + 1,
            action_names.get(sample.label).map(|s| s.as_str()).unwrap_or("알 수 없음"),
            action_names.get(pred_label as usize).map(|s| s.as_str()).unwrap_or("알 수 없음"),
            mark,
            latency
        );
        let prob_line = action_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let p = probs_vec.get(i).copied().unwrap_or(0.0) * 100.0;
                let short_name = name.split_whitespace().next().unwrap_or(name);
                format!("{}: {:.1}%", short_name, p)
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("   확률: {}", prob_line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_stabilizer_noise_to_rest_fast_transition() {
        let mut stab = GestureStabilizer::new(3, 45.0);
        // 강제로 Class 1 (동적 잡음) 상태로 설정하고 불응기 만료
        stab.current_stable_gesture = 1;
        stab.last_transition_time = Instant::now() - Duration::from_millis(500);

        // 휴식(0) 확률 유입
        let rest_probs = [0.90, 0.05, 0.05];
        let d1 = stab.update(&rest_probs, 20.0);
        assert_eq!(d1.state_tag, "전이");
        assert_eq!(d1.stable_idx, 1);
        assert_eq!(d1.candidate_count, 1);
        assert_eq!(d1.required_debounce, 2);

        // 2번째 프레임에 즉시 휴식(0) 확정! (기존 12프레임 갇힘 문제 완벽 해결)
        let d2 = stab.update(&rest_probs, 20.0);
        assert_eq!(d2.state_tag, "확정");
        assert_eq!(d2.stable_idx, 0);
    }

    #[test]
    fn test_stabilizer_rest_to_noise_debouncing() {
        let mut stab = GestureStabilizer::new(3, 45.0);
        stab.current_stable_gesture = 0;
        stab.last_transition_time = Instant::now() - Duration::from_millis(500);

        let noise_probs = [0.05, 0.90, 0.05];
        // Frame 1: EMA 필터링 중 (아직 EMA에서는 0이 우세 -> 안정 유지)
        let d1 = stab.update(&noise_probs, 25.0);
        assert_eq!(d1.state_tag, "안정");
        assert_eq!(d1.stable_idx, 0);

        // Frame 2..5: EMA가 1로 전환된 후 디바운서 1/5 ~ 4/5 누적
        for i in 1..=4 {
            let d = stab.update(&noise_probs, 25.0);
            assert_eq!(d.state_tag, "전이");
            assert_eq!(d.stable_idx, 0);
            assert_eq!(d.candidate_count, i);
            assert_eq!(d.required_debounce, 5);
        }

        // Frame 6: 5프레임 검증 완료 -> 잡음(1) 확정
        let d6 = stab.update(&noise_probs, 25.0);
        assert_eq!(d6.state_tag, "확정");
        assert_eq!(d6.stable_idx, 1);
    }

    #[test]
    fn test_stabilizer_fist_sustained_noise_defense() {
        let mut stab = GestureStabilizer::new(3, 45.0);
        // 진짜 주먹(2) 수축 중 (EMA도 주먹으로 초기화)
        stab.current_stable_gesture = 2;
        stab.ema_probs = vec![0.0, 0.0, 1.0];
        stab.last_transition_time = Instant::now() - Duration::from_millis(500);

        // 잡음(1) 신호가 지속되어 EMA 상으로 잡음이 우세해져도(Cand 1), 수축 유지 에너지(RMS 120.0)가 있으면 잡음 방어로 악력 유지
        let noise_probs = [0.05, 0.90, 0.05];
        // EMA 전환을 위해 몇 프레임 투입
        let _ = stab.update(&noise_probs, 120.0);
        let _ = stab.update(&noise_probs, 120.0);
        let d = stab.update(&noise_probs, 120.0);
        assert_eq!(d.state_tag, "유지");
        assert_eq!(d.stable_idx, 2);
        assert!(d.reason.contains("수축 유지 중 잡음 방어"));
    }

    #[test]
    fn test_stabilizer_fist_to_rest_release() {
        let mut stab = GestureStabilizer::new(3, 45.0);
        stab.current_stable_gesture = 2;
        stab.ema_probs = vec![0.0, 0.0, 1.0];
        stab.last_transition_time = Instant::now() - Duration::from_millis(500);

        let rest_probs = [0.90, 0.05, 0.05];
        // Frame 1: EMA 전환 중
        let _ = stab.update(&rest_probs, 20.0);
        // EMA 상으로 0이 우세해진 후 디바운싱 검증
        let d1 = stab.update(&rest_probs, 20.0);
        assert_eq!(d1.best_cand, 0);
        assert_eq!(d1.required_debounce, 5);
    }

    #[test]
    fn test_stabilizer_multi_class_extension() {
        // 4클래스: 0=휴식, 1=잡음, 2=주먹, 3=손펴기
        let mut stab = GestureStabilizer::new(4, 45.0);
        stab.current_stable_gesture = 2; // 주먹
        stab.ema_probs = vec![0.0, 0.0, 1.0, 0.0];
        stab.last_transition_time = Instant::now() - Duration::from_millis(500);

        // 주먹(2) -> 손펴기(3) 전이 (능동 수축 에너지 RMS 150.0)
        let open_probs = [0.05, 0.05, 0.05, 0.85];
        // EMA가 3으로 역전되어 50%를 초과할 때까지 2프레임 주입
        let _ = stab.update(&open_probs, 150.0);
        let _ = stab.update(&open_probs, 150.0);

        let d1 = stab.update(&open_probs, 150.0);
        assert_eq!(d1.best_cand, 3);
        assert_eq!(d1.required_debounce, 4); // 제스처 간 전환: 4프레임
        assert_eq!(d1.candidate_count, 1);

        let _ = stab.update(&open_probs, 150.0); // count = 2
        let _ = stab.update(&open_probs, 150.0); // count = 3
        let d4 = stab.update(&open_probs, 150.0); // count = 4 -> 확정!
        assert_eq!(d4.state_tag, "확정");
        assert_eq!(d4.stable_idx, 3);

        // 손펴기(3) 수축 유지 중 잡음(1) 방어 (수축 유지 에너지 RMS 120.0)
        stab.last_transition_time = Instant::now() - Duration::from_millis(500);
        let noise_probs = [0.05, 0.90, 0.02, 0.03];
        let _ = stab.update(&noise_probs, 120.0);
        let _ = stab.update(&noise_probs, 120.0);
        let dn = stab.update(&noise_probs, 120.0);
        assert_eq!(dn.state_tag, "유지");
        assert_eq!(dn.stable_idx, 3); // 손펴기 악력 유지
        assert!(dn.reason.contains("수축 유지 중 잡음 방어"));
    }

    #[test]
    fn test_stabilizer_noise_to_fist_blocked_by_energy_gate() {
        let mut stab = GestureStabilizer::new(3, 45.0);
        stab.current_stable_gesture = 1; // 동적 잡음 상태
        stab.last_transition_time = Instant::now() - Duration::from_millis(500);

        // 신경망이 일시적으로 주먹(2)을 95%로 예측했으나 실제 RMS가 70.0 (잡음 수준, < 110.0)인 경우
        let fist_spike_probs = [0.02, 0.03, 0.95];
        for _ in 0..10 {
            let d = stab.update(&fist_spike_probs, 70.0);
            // 물리 인터락에 의해 Class 1로 강제 리매핑되어 주먹(2)으로 절대 전이되지 않음!
            assert_eq!(d.stable_idx, 1);
            assert_ne!(d.best_cand, 2);
        }
    }

    #[test]
    fn test_stabilizer_noise_to_fist_debouncing_from_noise() {
        let mut stab = GestureStabilizer::new(3, 45.0);
        stab.current_stable_gesture = 1; // 동적 잡음 상태
        stab.last_transition_time = Instant::now() - Duration::from_millis(500);

        // 진짜 주먹 수축: RMS 250.0 (에너지 게이트 통과), 확률 90%
        let fist_probs = [0.02, 0.08, 0.90];
        // Frame 1: EMA 전환
        let _ = stab.update(&fist_probs, 250.0);
        let d = stab.update(&fist_probs, 250.0);
        assert_eq!(d.best_cand, 2);
        // 동적 잡음(1) -> 주먹(2) 진입은 3프레임이 아닌 5프레임 엄격 검증 요구!
        assert_eq!(d.required_debounce, 5);
    }

    #[test]
    fn test_stabilizer_fist_noise_decay_release_when_relaxed() {
        let mut stab = GestureStabilizer::new(3, 45.0);
        stab.current_stable_gesture = 2; // 주먹 상태
        stab.ema_probs = vec![0.0, 0.0, 1.0];
        stab.last_transition_time = Instant::now() - Duration::from_millis(500);

        // 사용자가 손을 펴서 RMS가 50.0(< min_sustain_rms 80.0)으로 떨어지며 잡음(1) 유입
        let noise_probs = [0.05, 0.90, 0.05];
        let _ = stab.update(&noise_probs, 50.0); // EMA 전이
        let d1 = stab.update(&noise_probs, 50.0);
        assert_eq!(d1.best_cand, 1);
        assert_eq!(d1.state_tag, "전이"); // 잡음 방어에 갇히지 않고 전이 시작!

        let _ = stab.update(&noise_probs, 50.0);
        let d3 = stab.update(&noise_probs, 50.0);
        assert_eq!(d3.state_tag, "확정");
        assert_eq!(d3.stable_idx, 1); // 3프레임 만에 동적 잡음(1)으로 정상 탈출!
    }
}


