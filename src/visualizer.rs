use glob::glob;
use plotly::color::Rgba;
use plotly::common::{Marker, Mode, Title};
use plotly::layout::{Axis, Layout, LayoutScene};
use plotly::{Plot, Scatter3D};
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// CSV 데이터셋의 1개 샘플 레코드 구조체
#[derive(Debug, Clone)]
pub struct EmgCsvRecord {
    pub index: usize,
    pub features: Vec<f32>,
    pub label: usize,
    pub action_name: String,
    pub total_ac_rms: f32,
    pub total_wl: f32,
    pub dominant_fft_power: f32,
    pub ch_rmss: [f32; 5],
}

/// 3D 시각화 모드
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationMode {
    /// 75차원 전체 특징을 PCA를 통해 상위 3개 주성분 축(PC1, PC2, PC3)으로 직교 투영
    Pca3D,
    /// 물리적 의미가 명확한 3대 핵심 특징(총 RMS, 파형 길이 WL, 우세 주파수 진폭) 3D 산점도
    PhysicalFeatures3D,
    /// 시간(프레임) × 5개 센서 채널 × AC-RMS 에너지의 시계열 3D 궤적 리본 뷰
    TimeTrajectory3D,
    /// 3가지 3D 그래프 모두 일괄 생성 및 브라우저 오픈
    All,
}

/// 주성분 분석 (PCA)을 통한 고차원 특징 공간 3차원 투영
pub fn compute_pca_3d(data: &[Vec<f32>]) -> (Vec<[f32; 3]>, [f32; 3]) {
    let n = data.len();
    if n == 0 {
        return (Vec::new(), [0.0; 3]);
    }
    let d = data[0].len();
    if d < 3 {
        let mut proj = Vec::with_capacity(n);
        for row in data {
            let mut p = [0.0; 3];
            for (i, val) in row.iter().enumerate().take(3) {
                p[i] = *val;
            }
            proj.push(p);
        }
        return (proj, [1.0, 1.0, 1.0]);
    }

    // 1. 각 특징 차원별 평균(Mean) 및 표준편차(Std) 산출 (Z-Score 표준화)
    let mut means = vec![0.0f32; d];
    let mut stds = vec![0.0f32; d];

    for row in data {
        for j in 0..d {
            means[j] += row[j];
        }
    }
    for j in 0..d {
        means[j] /= n as f32;
    }

    for row in data {
        for j in 0..d {
            let diff = row[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    for j in 0..d {
        stds[j] = (stds[j] / n as f32).sqrt().max(1e-5);
    }

    let mut z = Vec::with_capacity(n);
    for row in data {
        let mut z_row = Vec::with_capacity(d);
        for j in 0..d {
            z_row.push((row[j] - means[j]) / stds[j]);
        }
        z.push(z_row);
    }

    // 2. 공분산 행렬 C = (1/N) * Z^T * Z 계산
    let mut cov = vec![vec![0.0f32; d]; d];
    for i in 0..n {
        for j in 0..d {
            let zj = z[i][j];
            for k in j..d {
                cov[j][k] += zj * z[i][k];
            }
        }
    }
    for j in 0..d {
        cov[j][j] /= n as f32;
        for k in (j + 1)..d {
            cov[j][k] /= n as f32;
            cov[k][j] = cov[j][k];
        }
    }

    // 3. Power Iteration 및 Deflation을 통한 상위 3개 고유벡터/고유값 산출
    let mut eigenvectors = Vec::new();
    let mut eigenvalues = [0.0f32; 3];
    let mut current_cov = cov;

    for comp in 0..3 {
        let mut v = vec![0.0f32; d];
        for (idx, item) in v.iter_mut().enumerate() {
            *item = ((idx + comp + 1) as f32 * 0.71).sin();
        }
        let init_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-5);
        for item in &mut v {
            *item /= init_norm;
        }

        for _ in 0..40 {
            let mut w = vec![0.0f32; d];
            for j in 0..d {
                let mut sum = 0.0f32;
                for k in 0..d {
                    sum += current_cov[j][k] * v[k];
                }
                w[j] = sum;
            }
            let norm = w.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-7);
            for j in 0..d {
                v[j] = w[j] / norm;
            }
        }

        let mut lambda = 0.0f32;
        for j in 0..d {
            let mut row_sum = 0.0f32;
            for k in 0..d {
                row_sum += current_cov[j][k] * v[k];
            }
            lambda += v[j] * row_sum;
        }
        eigenvalues[comp] = lambda.max(0.0);

        for j in 0..d {
            for k in 0..d {
                current_cov[j][k] -= lambda * v[j] * v[k];
            }
        }

        eigenvectors.push(v);
    }

    // 4. 각 샘플을 3차원 주성분 축으로 직교 투영
    let v1 = &eigenvectors[0];
    let v2 = &eigenvectors[1];
    let v3 = &eigenvectors[2];

    let mut projected = Vec::with_capacity(n);
    for row in &z {
        let p1: f32 = row.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
        let p2: f32 = row.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let p3: f32 = row.iter().zip(v3.iter()).map(|(a, b)| a * b).sum();
        projected.push([p1, p2, p3]);
    }

    (projected, eigenvalues)
}

/// CSV 데이터셋 파일을 읽어 파싱합니다.
pub fn load_csv_records(file_path: &Path) -> io::Result<Vec<EmgCsvRecord>> {
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut records = Vec::new();

    for (idx, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let len = record.len();
        if len < 4 {
            continue;
        }

        let features: Vec<f32> = (0..len - 2)
            .map(|i| record[i].parse::<f32>().unwrap_or(0.0))
            .collect();
        let label: usize = record[len - 2].parse().unwrap_or(0);
        let action_name = record[len - 1].to_string();

        // 75차원인 경우: [0..5 raw AC, 5..45 FFT, 45..75 시간도메인]
        // 시간도메인: 채널당 [RMS, MAV, WL, ZC, SSC, WAMP]
        let mut ch_rmss = [0.0f32; 5];
        let mut total_wl = 0.0f32;
        let mut dominant_fft_power = 0.0f32;

        if features.len() >= 75 {
            for ch in 0..5 {
                let rms_idx = 45 + ch * 6;
                let wl_idx = 45 + ch * 6 + 2;
                if rms_idx < features.len() {
                    ch_rmss[ch] = features[rms_idx];
                }
                if wl_idx < features.len() {
                    total_wl += features[wl_idx];
                }
            }
            // Ch0, Ch1 FFT 진폭 합산
            for bin in 0..16 {
                if 5 + bin < features.len() {
                    dominant_fft_power += features[5 + bin];
                }
            }
        } else {
            // 다른 차원 포맷일 경우 대비 fallback
            for ch in 0..5.min(features.len()) {
                ch_rmss[ch] = features[ch].abs();
            }
        }

        let total_ac_rms = (ch_rmss.iter().map(|&r| r * r).sum::<f32>()).sqrt();

        records.push(EmgCsvRecord {
            index: idx,
            features,
            label,
            action_name,
            total_ac_rms,
            total_wl,
            dominant_fft_power,
            ch_rmss,
        });
    }

    Ok(records)
}

/// 동작 클래스별 전용 컬러 반환
fn get_class_color(label: usize) -> (Rgba, &'static str) {
    match label {
        0 => (Rgba::new(46, 204, 113, 0.85), "#2ecc71"),  // Emerald Green (휴식)
        1 => (Rgba::new(231, 76, 60, 0.85), "#e74c3c"),   // Crimson Red (주먹 / 엄지)
        2 => (Rgba::new(52, 152, 219, 0.85), "#3498db"),  // Dodger Blue (손펴기 / 검지)
        3 => (Rgba::new(243, 156, 18, 0.85), "#f39c12"),  // Amber Orange (엄지 / 중지)
        4 => (Rgba::new(155, 89, 182, 0.85), "#9b59b6"),  // Amethyst Purple (약지)
        5 => (Rgba::new(26, 188, 156, 0.85), "#1abc9c"),  // Turquoise Teal (새끼)
        _ => (Rgba::new(149, 165, 166, 0.85), "#95a5a6"),  // Gray (기타)
    }
}

/// [1] PCA 3D 주성분 공간 투영 그래프 생성
pub fn generate_pca_3d_plot(records: &[EmgCsvRecord], dataset_name: &str) -> Plot {
    let feature_matrix: Vec<Vec<f32>> = records.iter().map(|r| r.features.clone()).collect();
    let (projected, eigenvalues) = compute_pca_3d(&feature_matrix);
    let total_var: f32 = eigenvalues.iter().sum::<f32>().max(1e-5);
    let pc1_ratio = (eigenvalues[0] / total_var) * 100.0;
    let pc2_ratio = (eigenvalues[1] / total_var) * 100.0;
    let pc3_ratio = (eigenvalues[2] / total_var) * 100.0;

    let mut plot = Plot::new();

    let num_classes = records.iter().map(|r| r.label).max().map(|m| m + 1).unwrap_or(2).max(2);
    for label_id in 0..num_classes {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut zs = Vec::new();
        let mut hover_texts = Vec::new();
        let mut class_name = String::new();

        for (idx, r) in records.iter().enumerate() {
            if r.label == label_id {
                if class_name.is_empty() {
                    class_name = r.action_name.clone();
                }
                xs.push(projected[idx][0]);
                ys.push(projected[idx][1]);
                zs.push(projected[idx][2]);

                hover_texts.push(format!(
                    "<b>샘플 #{}</b><br>동작: {}<br>PC1: {:.2}<br>PC2: {:.2}<br>PC3: {:.2}<br>Total RMS: {:.1}",
                    r.index, r.action_name, projected[idx][0], projected[idx][1], projected[idx][2], r.total_ac_rms
                ));
            }
        }

        if xs.is_empty() {
            continue;
        }

        let (rgba_color, _) = get_class_color(label_id);
        let trace = Scatter3D::new(xs, ys, zs)
            .mode(Mode::Markers)
            .name(&class_name)
            .marker(Marker::new().size(4).color(rgba_color))
            .hover_text_array(hover_texts);

        plot.add_trace(trace);
    }

    let title_text = format!(
        "🌟 sEMG 75차원 특징 공간 PCA 3D 주성분 분석 투영도 [{}]",
        dataset_name
    );
    let scene = LayoutScene::new()
        .x_axis(Axis::new().title(Title::with_text(format!("PC 1 ({:.1}%)", pc1_ratio))))
        .y_axis(Axis::new().title(Title::with_text(format!("PC 2 ({:.1}%)", pc2_ratio))))
        .z_axis(Axis::new().title(Title::with_text(format!("PC 3 ({:.1}%)", pc3_ratio))));

    let layout = Layout::new().title(Title::with_text(title_text)).scene(scene);
    plot.set_layout(layout);
    plot
}

/// [2] 3대 핵심 생체 물리 특징 3D 공간 그래프 생성
pub fn generate_physical_3d_plot(records: &[EmgCsvRecord], dataset_name: &str) -> Plot {
    let mut plot = Plot::new();

    let num_classes = records.iter().map(|r| r.label).max().map(|m| m + 1).unwrap_or(2).max(2);
    for label_id in 0..num_classes {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut zs = Vec::new();
        let mut hover_texts = Vec::new();
        let mut class_name = String::new();

        for r in records {
            if r.label == label_id {
                if class_name.is_empty() {
                    class_name = r.action_name.clone();
                }
                xs.push(r.total_ac_rms);
                ys.push(r.total_wl);
                zs.push(r.dominant_fft_power);

                hover_texts.push(format!(
                    "<b>샘플 #{}</b><br>동작: {}<br>총 AC-RMS: {:.1}<br>파형 길이(WL): {:.1}<br>우세 FFT 진폭: {:.1}",
                    r.index, r.action_name, r.total_ac_rms, r.total_wl, r.dominant_fft_power
                ));
            }
        }

        if xs.is_empty() {
            continue;
        }

        let (rgba_color, _) = get_class_color(label_id);
        let trace = Scatter3D::new(xs, ys, zs)
            .mode(Mode::Markers)
            .name(&class_name)
            .marker(Marker::new().size(4).color(rgba_color))
            .hover_text_array(hover_texts);

        plot.add_trace(trace);
    }

    let title_text = format!(
        "⚡ 3대 핵심 생체 물리 지표(RMS × WL × FFT) 3D 산점도 [{}]",
        dataset_name
    );
    let scene = LayoutScene::new()
        .x_axis(Axis::new().title(Title::with_text("총 교류 실효치 (Total AC-RMS)")))
        .y_axis(Axis::new().title(Title::with_text("총 파형 길이 (Waveform Length)")))
        .z_axis(Axis::new().title(Title::with_text("우세 주파수 진폭 (Dominant FFT Magnitude)")));

    let layout = Layout::new().title(Title::with_text(title_text)).scene(scene);
    plot.set_layout(layout);
    plot
}

/// [3] 3D 시계열 궤적 / 다채널 리본 그래프 생성
pub fn generate_trajectory_3d_plot(records: &[EmgCsvRecord], dataset_name: &str) -> Plot {
    let mut plot = Plot::new();

    // 채널 0부터 4까지 5개 채널의 시간 흐름에 따른 RMS 궤적 선 그래프
    for ch in 0..5 {
        let xs: Vec<f32> = records.iter().map(|r| r.index as f32).collect();
        let ys: Vec<f32> = records.iter().map(|_| ch as f32).collect();
        let zs: Vec<f32> = records.iter().map(|r| r.ch_rmss[ch]).collect();

        let hover_texts: Vec<String> = records
            .iter()
            .map(|r| {
                format!(
                    "<b>프레임 #{}</b><br>동작: {}<br>채널: Ch{}<br>RMS: {:.1}",
                    r.index, r.action_name, ch, r.ch_rmss[ch]
                )
            })
            .collect();

        let trace = Scatter3D::new(xs, ys, zs)
            .mode(Mode::LinesMarkers)
            .name(&format!("센서 Ch{}", ch))
            .marker(Marker::new().size(3))
            .hover_text_array(hover_texts);

        plot.add_trace(trace);
    }

    let title_text = format!(
        "⏱️ 다채널 sEMG 시계열 3차원 궤적(Time-Trajectory) 리본 뷰 [{}]",
        dataset_name
    );
    let scene = LayoutScene::new()
        .x_axis(Axis::new().title(Title::with_text("시간 (Frame Index)")))
        .y_axis(Axis::new().title(Title::with_text("센서 채널 (Ch 0 ~ 4)")))
        .z_axis(Axis::new().title(Title::with_text("채널별 AC-RMS 에너지")));

    let layout = Layout::new().title(Title::with_text(title_text)).scene(scene);
    plot.set_layout(layout);
    plot
}

/// 3차원 그래프 시각화 메뉴 인터랙티브 실행
pub fn run_3d_visualization_menu() {
    let mut found_files = Vec::new();
    for entry in glob("emg_dataset_*.csv").expect("Failed glob") {
        if let Ok(path) = entry {
            found_files.push(path);
        }
    }
    for entry in glob("mlp/*.csv").expect("Failed glob") {
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
        println!("\n❌ [에러] 시각화할 CSV 데이터셋 파일이 없습니다. 1번 데이터 수집을 먼저 실행해주세요.");
        return;
    }

    println!("\n========================================================");
    println!(" 📈 Plotly 3차원 데이터셋 시각화 (3D Interactive Graph)");
    println!("========================================================");
    println!("[시각화할 데이터셋 파일 선택]");
    for (i, file) in found_files.iter().enumerate() {
        println!("  [{}] {}", i + 1, file.display());
    }
    print!("선택 번호 (기본 1) > ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let choice = input.trim().parse::<usize>().unwrap_or(1);
    let dataset_path: &PathBuf = if choice > 0 && choice <= found_files.len() {
        &found_files[choice - 1]
    } else {
        &found_files[0]
    };

    println!("\n[*] 데이터셋 로드 중: {}...", dataset_path.display());
    let records = match load_csv_records(dataset_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("❌ CSV 로드 실패: {}", e);
            return;
        }
    };
    println!("✔ 총 {}개의 샘플 데이터를 성공적으로 읽었습니다.", records.len());

    println!("\n[3차원 그래프 시각화 모드 선택]");
    println!("  [1] 🌟 PCA 3D 주성분 공간 투영도 (🥇 추천! 75차원 전체 특징 클러스터 분석)");
    println!("  [2] ⚡ 3대 핵심 생체 물리 특징 3D 공간 (총 RMS × 파형길이 WL × 우세 FFT)");
    println!("  [3] ⏱️ 시계열 3D 궤적 리본 뷰 (시간 경과에 따른 5채널 신호 파형 흐름)");
    println!("  [4] 🚀 3가지 3D 그래프 모두 일괄 생성 및 브라우저 열기");
    print!("선택 (1-4, 기본 1) > ");
    io::stdout().flush().unwrap();

    let mut mode_input = String::new();
    io::stdin().read_line(&mut mode_input).unwrap();
    let mode = match mode_input.trim() {
        "2" => VisualizationMode::PhysicalFeatures3D,
        "3" => VisualizationMode::TimeTrajectory3D,
        "4" => VisualizationMode::All,
        _ => VisualizationMode::Pca3D,
    };

    let dataset_file_name = dataset_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("dataset.csv");

    match mode {
        VisualizationMode::Pca3D => {
            println!("\n🎨 PCA 3D 주성분 투영 그래프를 연산 및 렌더링 중...");
            let plot = generate_pca_3d_plot(&records, dataset_file_name);
            let html_file = "emg_3d_pca.html";
            plot.show_html(html_file);
            println!("✔ 3D 그래프가 생성되어 기본 웹 브라우저에서 열렸습니다!");
            println!("   📂 저장된 파일: file://{}", Path::new(html_file).canonicalize().unwrap_or(PathBuf::from(html_file)).display());
        }
        VisualizationMode::PhysicalFeatures3D => {
            println!("\n🎨 생체 물리 특징 3D 산점도 그래프를 연산 및 렌더링 중...");
            let plot = generate_physical_3d_plot(&records, dataset_file_name);
            let html_file = "emg_3d_physical.html";
            plot.show_html(html_file);
            println!("✔ 3D 그래프가 생성되어 기본 웹 브라우저에서 열렸습니다!");
            println!("   📂 저장된 파일: file://{}", Path::new(html_file).canonicalize().unwrap_or(PathBuf::from(html_file)).display());
        }
        VisualizationMode::TimeTrajectory3D => {
            println!("\n🎨 시계열 3D 궤적 리본 그래프를 연산 및 렌더링 중...");
            let plot = generate_trajectory_3d_plot(&records, dataset_file_name);
            let html_file = "emg_3d_trajectory.html";
            plot.show_html(html_file);
            println!("✔ 3D 그래프가 생성되어 기본 웹 브라우저에서 열렸습니다!");
            println!("   📂 저장된 파일: file://{}", Path::new(html_file).canonicalize().unwrap_or(PathBuf::from(html_file)).display());
        }
        VisualizationMode::All => {
            println!("\n🎨 3가지 3D 그래프를 일괄 렌더링 중...");
            let p1 = generate_pca_3d_plot(&records, dataset_file_name);
            p1.show_html("emg_3d_pca.html");
            let p2 = generate_physical_3d_plot(&records, dataset_file_name);
            p2.show_html("emg_3d_physical.html");
            let p3 = generate_trajectory_3d_plot(&records, dataset_file_name);
            p3.show_html("emg_3d_trajectory.html");

            println!("✔ 3가지 3D 그래프가 모두 생성되어 브라우저에서 열렸습니다!");
            println!("   1) PCA 주성분 투영: emg_3d_pca.html");
            println!("   2) 생체 물리 지표:   emg_3d_physical.html");
            println!("   3) 시계열 궤적:     emg_3d_trajectory.html");
        }
    }
}
