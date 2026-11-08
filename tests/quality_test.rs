use emg_ml_pipeline::quality::{DatasetQualityReport, QualityGrade};

#[test]
fn test_quality_evaluator_on_datasets() {
    // 1. 정상 데이터셋 테스트 (20260922_181716.csv)
    let good_path = "emg_dataset_20260922_181716.csv";
    if std::path::Path::new(good_path).exists() {
        let rep = DatasetQualityReport::analyze(good_path).expect("Failed to analyze good dataset");
        println!("\n=== [Good Dataset Test] ===");
        rep.print_report();
        assert!(rep.total_score >= 70.0, "Good dataset should score >= 70");
        assert_ne!(rep.grade, QualityGrade::GradeF, "Good dataset should not be Grade F");
    }

    // 3. 노이즈 내성 3대 동작 데이터셋 테스트 (emg_dataset_noise_robust_20261002_152455.csv)
    let noise_robust_path = "emg_dataset_noise_robust_20261002_152455.csv";
    if std::path::Path::new(noise_robust_path).exists() {
        let rep = DatasetQualityReport::analyze(noise_robust_path).expect("Failed to analyze noise robust dataset");
        println!("\n=== [Noise Robust Dataset Test] ===");
        rep.print_report();
    }
}
