use burn_wgpu::WgpuDevice;
use emg_ml_pipeline::{train_model, ModelArchitecture, TrainingConfig, TrainingProgress};
use std::path::Path;

#[test]
fn test_train_all_three_models() {
    let device = WgpuDevice::default();
    let dataset_path = "emg_dataset_20260919_173715.csv";
    
    if !Path::new(dataset_path).exists() {
        println!("데이터셋 파일이 없어 테스트를 건너뜁니다: {}", dataset_path);
        return;
    }

    let archs = [
        ModelArchitecture::Cnn,
        ModelArchitecture::Lstm,
        ModelArchitecture::CnnLstm,
    ];

    for arch in archs {
        println!("Testing training for {:?}...", arch);
        let config = TrainingConfig {
            dataset_path: dataset_path.to_string(),
            model_arch: arch,
            epochs: 2, // 검증용 2 epoch
            learning_rate: 1e-2,
            batch_size: 32,
        };

        let result = train_model(&device, &config, |prog: TrainingProgress| {
            println!("  Epoch: {}/{}, loss: {}", prog.epoch, prog.total_epochs, prog.loss);
        });

        assert!(result.is_ok(), "Training failed for {:?}", arch);
        let saved_file = result.unwrap();
        assert!(Path::new(&saved_file).exists(), "Model file not found: {}", saved_file);
        println!("✔ {:?} trained successfully, saved to {}", arch, saved_file);
    }
}
