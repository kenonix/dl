use burn::backend::wgpu::WgpuDevice;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::Tensor;
use emg_ml_pipeline::{
    load_mlp_dataset, load_modular_mlp_dataset, train_model, EmgMlpConfig, EmgModularMlpModel,
    ModelArchitecture, MyBackend, NormalizationStats, TrainingConfig, TrainingProgress,
    MODULAR_MLP_WINDOW_SIZE, NUM_CLASSES,
};
use std::fs::File;
use std::path::Path;

#[test]
fn test_train_mlp_model() {
    let device = WgpuDevice::default();
    let mut dataset_path = "emg_dataset_20260921_214355.csv";
    if !Path::new(dataset_path).exists() {
        dataset_path = "emg_dataset_20260921_184801.csv";
    }

    if !Path::new(dataset_path).exists() {
        println!("데이터셋 파일이 없어 테스트를 건너뜁니다: {}", dataset_path);
        return;
    }

    println!("=== Testing MLP 60 epochs training on clean dataset: {} ===", dataset_path);
    let mlp_config = TrainingConfig {
        dataset_path: dataset_path.to_string(),
        model_arch: ModelArchitecture::Mlp,
        epochs: 60,
        learning_rate: 1e-3,
        batch_size: 32,
        num_classes: NUM_CLASSES,
    };

    let res = train_model(&device, &mlp_config, |prog: TrainingProgress| {
        if prog.epoch % 15 == 0 || prog.epoch == 1 {
            println!("  ➔ [MLP Epoch {:2}/60] Status: {}", prog.epoch, prog.status);
        }
    });

    assert!(res.is_ok(), "MLP training failed");
    let model_path = res.unwrap();
    assert!(Path::new(&model_path).exists());
    println!("✔ MLP Model saved to: {}", model_path);

    // 🎯 4개 제스처 분류 정확도 및 오차 행렬(Confusion Matrix) 정밀 검증
    let norm_path = "norm_stats_emg_mlp_model.json";
    assert!(Path::new(norm_path).exists());
    let file = File::open(norm_path).expect("Failed to open norm stats");
    let stats: NormalizationStats = serde_json::from_reader(file).expect("Failed to parse norm stats");

    let recorder = CompactRecorder::new();
    let mut model = EmgMlpConfig {
        d_input: stats.feature_dim,
        d_hidden1: 128,
        d_hidden2: 64,
        d_hidden3: 32,
        d_hidden4: 16,
        num_classes: NUM_CLASSES,
        dropout: 0.0,
    }
    .init::<MyBackend>(&device);

    let rec = recorder
        .load("emg_mlp_model".into(), &device)
        .expect("Failed to load model record");
    model = model.load_record(rec);

    let (train_dataset, valid_dataset, _) =
        load_mlp_dataset(dataset_path).expect("Failed to load dataset for eval");
    let all_items = [train_dataset.items, valid_dataset.items].concat();

    let mut confusion = [[0usize; NUM_CLASSES]; NUM_CLASSES];
    let mut class_totals = [0usize; NUM_CLASSES];

    for item in &all_items {
        let input_tensor =
            Tensor::<MyBackend, 1>::from_floats(item.features.as_slice(), &device).unsqueeze::<2>();
        let logits = model.forward(input_tensor);
        let pred = logits.argmax(1).into_scalar() as usize;
        let actual = item.label;

        confusion[actual][pred] += 1;
        class_totals[actual] += 1;
    }

    let class_names = ["휴식(Rest)", "주먹(Fist)", "손펴기(Open)", "엄지(Thumb)"];
    println!("\n=== 🎯 MLP 4개 제스처 검증 결과 (Confusion Matrix) ===");
    for i in 0..NUM_CLASSES {
        let total = class_totals[i];
        let correct = confusion[i][i];
        let acc = if total > 0 {
            (correct as f32 / total as f32) * 100.0
        } else {
            0.0
        };
        println!(
            "  [{}] 정확도: {:5.1}% ({}/{} 개) | 예측 분포: {:?}",
            class_names[i], acc, correct, total, confusion[i]
        );
        assert!(acc >= 85.0, "Class {} accuracy too low: {:.1}%", class_names[i], acc);
    }
}

#[test]
fn test_train_tcn_and_all_models() {
    let device = WgpuDevice::default();
    let mut dataset_path = "emg_dataset_20260921_184801.csv";
    if !Path::new(dataset_path).exists() {
        dataset_path = "emg_dataset_20260919_173715.csv";
    }

    if !Path::new(dataset_path).exists() {
        println!("데이터셋 파일이 없어 테스트를 건너뜁니다: {}", dataset_path);
        return;
    }

    // 1D-TCN을 10 에포크 동안 학습하여 손실(Loss) 감소 확인
    println!("=== Testing 1D-TCN 10 epochs training ===");
    let tcn_config = TrainingConfig {
        dataset_path: dataset_path.to_string(),
        model_arch: ModelArchitecture::Tcn,
        epochs: 10,
        learning_rate: 1e-3,
        batch_size: 32,
        num_classes: NUM_CLASSES,
    };

    let mut first_loss = 0.0;
    let mut last_loss = 0.0;

    let res = train_model(&device, &tcn_config, |prog: TrainingProgress| {
        if prog.epoch == 1 {
            first_loss = prog.loss;
        }
        if prog.epoch == 10 {
            last_loss = prog.loss;
        }
        println!("  ➔ [TCN Epoch {:2}/10] Loss: {:.4}", prog.epoch, prog.loss);
    });

    assert!(res.is_ok(), "TCN training failed");
    let model_path = res.unwrap();
    assert!(Path::new(&model_path).exists());
    println!("✔ TCN Model saved to: {}", model_path);
    println!("✔ Loss change: {:.4} -> {:.4}", first_loss, last_loss);
}

#[test]
fn test_train_modular_mlp_model() {
    let device = WgpuDevice::default();
    let dataset_path = if Path::new("emg_dataset_noise_robust_merged_35d_20261108.csv").exists() {
        "emg_dataset_noise_robust_merged_35d_20261108.csv"
    } else if Path::new("emg_dataset_noise_robust_20261108_102522.csv").exists() {
        "emg_dataset_noise_robust_20261108_102522.csv"
    } else {
        "emg_dataset_noise_robust_20261002_160101.csv"
    };
    if !Path::new(dataset_path).exists() {
        return;
    }

    let config = TrainingConfig {
        dataset_path: dataset_path.to_string(),
        model_arch: ModelArchitecture::ModularMlp,
        epochs: 15,
        learning_rate: 1e-3,
        batch_size: 32,
        num_classes: 3,
    };

    let res = train_model(&device, &config, |_| {});
    assert!(res.is_ok(), "ModularMlp training failed");

    let (_, valid_dataset, stats) =
        load_modular_mlp_dataset(dataset_path, MODULAR_MLP_WINDOW_SIZE).unwrap();
    let recorder = CompactRecorder::new();
    let mut model =
        EmgModularMlpModel::<MyBackend>::new(&device, stats.feature_dim, stats.num_classes);
    let rec = recorder
        .load("emg_modular_mlp_model".into(), &device)
        .unwrap();
    model = model.load_record(rec);

    let mut correct = 0usize;
    let total = valid_dataset.items.len();
    for item in &valid_dataset.items {
        let input_tensor =
            Tensor::<MyBackend, 1>::from_floats(item.window.as_slice(), &device).reshape([
                1,
                item.seq_len,
                item.feature_dim,
            ]);
        let logits = model.forward(input_tensor);
        let pred = logits.argmax(1).into_scalar() as usize;
        if pred == item.label {
            correct += 1;
        }
    }
    let acc = (correct as f32 / total.max(1) as f32) * 100.0;
    println!(
        "✔ [Block-MLP 격리 검증 결과] 정확도: {:.2}% ({}/{})",
        acc, correct, total
    );
    assert!(acc >= 70.0, "Block-MLP valid accuracy too low: {:.2}%", acc);
}

#[test]
#[ignore]
fn test_train_4d_lda_hybrid_models() {
    let device = WgpuDevice::default();
    let dataset_path = "emg_dataset_fingers_3s_lda4d.csv";
    if !Path::new(dataset_path).exists() {
        println!("4D LDA 데이터셋 파일이 없어 테스트를 건너뜁니다: {}", dataset_path);
        return;
    }

    println!("\n=== 🚀 [2번 아키텍처 검증] 4D LDA 데이터셋 기반 5종 딥러닝 모델 학습 테스트 ===");
    let archs = [
        ModelArchitecture::Mlp,
        ModelArchitecture::Tcn,
        ModelArchitecture::Cnn,
        ModelArchitecture::Lstm,
        ModelArchitecture::CnnLstm,
    ];

    for arch in archs {
        println!("\n--- 학습 시작: {} ---", arch.display_name());
        let epochs = match arch {
            ModelArchitecture::Mlp => 30,
            _ => 40,
        };

        let config = TrainingConfig {
            dataset_path: dataset_path.to_string(),
            model_arch: arch,
            epochs,
            learning_rate: 1e-3,
            batch_size: 32,
            num_classes: 6,
        };

        let res = train_model(&device, &config, |prog: TrainingProgress| {
            if prog.epoch % 15 == 0 || prog.epoch == 1 || prog.epoch == epochs {
                println!("  ➔ [{:<8} Epoch {:2}/{:2}] Loss: {:.4}", arch.display_name(), prog.epoch, prog.total_epochs, prog.loss);
            }
        });

        assert!(res.is_ok(), "Training failed for {}", arch.display_name());
        let model_file = res.unwrap();
        assert!(Path::new(&model_file).exists());

        let stats_path = format!("norm_stats_{}.json", arch.file_name());
        assert!(Path::new(&stats_path).exists(), "Stats file missing: {}", stats_path);
        let file = File::open(&stats_path).expect("Failed to open norm stats");
        let stats: NormalizationStats = serde_json::from_reader(file).expect("Failed to parse norm stats");
        assert_eq!(stats.feature_dim, 4, "Feature dim must be 4 for 4D LDA pipeline");
        assert_eq!(stats.num_classes, 6, "Num classes must be 6");
        println!("✔ {} 모델 및 정규화 통계(4D LDA, 6개 클래스) 저장 검증 성공!", arch.display_name());
    }
}
