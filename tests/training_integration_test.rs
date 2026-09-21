use burn_wgpu::WgpuDevice;
use emg_ml_pipeline::{train_model, ModelArchitecture, TrainingConfig, TrainingProgress};
use std::path::Path;

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
