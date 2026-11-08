use emg_ml_pipeline::{
    compute_pca_3d, generate_pca_3d_plot, generate_physical_3d_plot,
    generate_trajectory_3d_plot, load_csv_records,
};
use std::path::Path;

#[test]
fn test_visualizer_csv_and_3d_plots() {
    let dataset_path = Path::new("emg_dataset_20260921_214355.csv");
    if !dataset_path.exists() {
        println!("CSV 파일이 없어 테스트를 건너뜁니다: {}", dataset_path.display());
        return;
    }

    let records = load_csv_records(dataset_path).expect("Failed to load records");
    assert!(!records.is_empty(), "Records should not be empty");
    println!("✔ 로드된 레코드 수: {}", records.len());

    // 1. PCA 3D 연산 검증
    let features: Vec<Vec<f32>> = records.iter().map(|r| r.features.clone()).collect();
    let (projected, eigenvalues) = compute_pca_3d(&features);
    assert_eq!(projected.len(), records.len());
    assert!(eigenvalues[0] > 0.0);
    assert!(eigenvalues[1] > 0.0);
    assert!(eigenvalues[2] > 0.0);
    println!("✔ PCA 고유값: PC1={:.2}, PC2={:.2}, PC3={:.2}", eigenvalues[0], eigenvalues[1], eigenvalues[2]);

    // 2. PCA 3D 플롯 HTML 생성 검증 및 파일 저장
    let pca_plot = generate_pca_3d_plot(&records, "emg_dataset_20260921_214355.csv");
    let pca_html = pca_plot.to_html();
    assert!(pca_html.contains("plotly"));
    pca_plot.write_html("emg_3d_pca.html");
    println!("✔ PCA 3D HTML 저장 완료 -> emg_3d_pca.html (길이: {} bytes)", pca_html.len());

    // 3. 생체 물리 지표 3D 플롯 검증 및 파일 저장
    let phys_plot = generate_physical_3d_plot(&records, "emg_dataset_20260921_214355.csv");
    let phys_html = phys_plot.to_html();
    assert!(phys_html.contains("plotly"));
    phys_plot.write_html("emg_3d_physical.html");
    println!("✔ 생체 물리 3D HTML 저장 완료 -> emg_3d_physical.html (길이: {} bytes)", phys_html.len());

    // 4. 시계열 궤적 3D 플롯 검증 및 파일 저장
    let traj_plot = generate_trajectory_3d_plot(&records, "emg_dataset_20260921_214355.csv");
    let traj_html = traj_plot.to_html();
    assert!(traj_html.contains("plotly"));
    traj_plot.write_html("emg_3d_trajectory.html");
    println!("✔ 시계열 3D 궤적 HTML 저장 완료 -> emg_3d_trajectory.html (길이: {} bytes)", traj_html.len());
}
