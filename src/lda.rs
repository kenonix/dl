//! Linear Discriminant Analysis (LDA) 모듈
//!
//! 고차원(75차원) EMG 특징 벡터를 6개 클래스(휴식 및 다섯 손가락) 간의
//! 분별력(Between-class scatter / Within-class scatter)을 최대화하는
//! 최적의 4차원 판별 공간(LD1, LD2, LD3, LD4)으로 직교 투영하고 분류합니다.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// LDA 모델 구조체
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LdaModel {
    /// 원본 특징 차원 (예: 75)
    pub feature_dim: usize,
    /// 축소 대상 차원 (예: 4)
    pub n_components: usize,
    /// 클래스 수 (예: 6)
    pub num_classes: usize,
    /// 전체 데이터 평균 벡터 [feature_dim]
    pub overall_mean: Vec<f32>,
    /// 클래스별 평균 벡터 [num_classes][feature_dim]
    pub class_means: Vec<Vec<f32>>,
    /// 각 클래스별 샘플 수
    pub class_counts: Vec<usize>,
    /// 투영 행렬 W [n_components][feature_dim] (각 행이 판별 벡터)
    pub projection_matrix: Vec<Vec<f32>>,
    /// 각 판별 축의 고유값 (분별력 척도)
    pub eigenvalues: Vec<f32>,
    /// 누적 분별 기여율 (Explained Variance Ratio, 백분율)
    pub explained_variance_ratios: Vec<f32>,
    /// 4차원 투영 공간에서의 클래스별 중심점 [num_classes][n_components]
    pub projected_class_centroids: Vec<Vec<f32>>,
}

impl LdaModel {
    /// 훈련 데이터 X [N, D] 와 라벨 y [N] 로부터 LDA 모델을 학습합니다.
    pub fn fit(x: &[Vec<f32>], y: &[usize], n_components: usize) -> Result<Self, String> {
        let n = x.len();
        if n == 0 {
            return Err("데이터가 비어 있습니다.".to_string());
        }
        let d = x[0].len();
        let max_label = y.iter().copied().max().unwrap_or(0);
        let num_classes = max_label + 1;

        if num_classes < 2 {
            return Err("최소 2개 이상의 클래스가 필요합니다.".to_string());
        }
        let target_dim = n_components.min(num_classes - 1).min(d);

        // 1. 전체 평균 및 클래스별 평균, 샘플 수 계산
        let mut overall_mean = vec![0.0f32; d];
        let mut class_means = vec![vec![0.0f32; d]; num_classes];
        let mut class_counts = vec![0usize; num_classes];

        for i in 0..n {
            let label = y[i];
            class_counts[label] += 1;
            for j in 0..d {
                let val = x[i][j];
                overall_mean[j] += val;
                class_means[label][j] += val;
            }
        }

        for j in 0..d {
            overall_mean[j] /= n as f32;
        }

        for c in 0..num_classes {
            let count = class_counts[c];
            if count == 0 {
                return Err(format!("클래스 {}의 샘플 수가 0입니다.", c));
            }
            for j in 0..d {
                class_means[c][j] /= count as f32;
            }
        }

        // 2. 클래스 내 산포 행렬 S_W 및 클래스 간 산포 행렬 S_B 계산
        let mut sw = vec![vec![0.0f32; d]; d];
        let mut sb = vec![vec![0.0f32; d]; d];

        // S_W = sum_c sum_{x in c} (x - mu_c)(x - mu_c)^T
        for i in 0..n {
            let c = y[i];
            let mu_c = &class_means[c];
            for j in 0..d {
                let diff_j = x[i][j] - mu_c[j];
                for k in j..d {
                    let diff_k = x[i][k] - mu_c[k];
                    let prod = diff_j * diff_k;
                    sw[j][k] += prod;
                }
            }
        }
        for j in 0..d {
            for k in (j + 1)..d {
                sw[k][j] = sw[j][k];
            }
        }

        // S_B = sum_c N_c (mu_c - mu)(mu_c - mu)^T
        for c in 0..num_classes {
            let n_c = class_counts[c] as f32;
            let mu_c = &class_means[c];
            for j in 0..d {
                let diff_j = mu_c[j] - overall_mean[j];
                for k in j..d {
                    let diff_k = mu_c[k] - overall_mean[k];
                    let prod = n_c * diff_j * diff_k;
                    sb[j][k] += prod;
                }
            }
        }
        for j in 0..d {
            for k in (j + 1)..d {
                sb[k][j] = sb[j][k];
            }
        }

        // 3. S_W 정규화 (특이 행렬 방지 및 Symmetric Positive Definite 보장)
        // alpha = 1e-4 * trace(S_W) / d
        let mut trace_sw = 0.0f32;
        for j in 0..d {
            trace_sw += sw[j][j];
        }
        let alpha = (1e-4f32 * (trace_sw / d as f32)).max(1e-6);
        for j in 0..d {
            sw[j][j] += alpha;
        }

        // 4. Cholesky 분해: S_W = L * L^T (L: 하삼각 행렬)
        let l = match cholesky_decompose(&sw, d) {
            Some(l_mat) => l_mat,
            None => {
                // 추가 댐핑 후 재시도
                for j in 0..d {
                    sw[j][j] += alpha * 10.0;
                }
                cholesky_decompose(&sw, d)
                    .ok_or_else(|| "Cholesky 분해에 실패했습니다. (행렬이 양의 정부호가 아님)".to_string())?
            }
        };

        // 5. L의 역행렬 L_inv 계산 (하삼각 행렬 전진 대입)
        let l_inv = invert_lower_triangular(&l, d);

        // 6. 대칭 행렬 M = L_inv * S_B * L_inv^T 계산
        // 먼저 Temp = S_B * L_inv^T = (L_inv * S_B)^T 계산
        // M[i][j] = sum_k sum_m L_inv[i][k] * S_B[k][m] * L_inv[j][m]
        let mut m_mat = vec![vec![0.0f32; d]; d];
        let mut temp = vec![vec![0.0f32; d]; d];

        for i in 0..d {
            for j in 0..d {
                let mut sum = 0.0f32;
                // L_inv는 하삼각 행렬이므로 k >= i인 경우에만 0이 아님
                for k in 0..=i {
                    sum += l_inv[i][k] * sb[k][j];
                }
                temp[i][j] = sum;
            }
        }

        for i in 0..d {
            for j in i..d {
                let mut sum = 0.0f32;
                for k in 0..=j {
                    sum += temp[i][k] * l_inv[j][k];
                }
                m_mat[i][j] = sum;
                m_mat[j][i] = sum;
            }
        }

        // 7. Jacobi 고유치 알고리즘으로 M의 모든 고유값 및 직교 고유벡터 계산
        let (eigenvalues_raw, v_mat) = jacobi_eigenvalue_sym(&m_mat, d, 60, 1e-7);

        // 8. 고유값 내림차순 정렬 인덱스 추출
        let mut eig_pairs: Vec<(usize, f32)> = eigenvalues_raw
            .into_iter()
            .enumerate()
            .map(|(idx, val)| (idx, val.max(0.0)))
            .collect();
        eig_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let total_variance: f32 = eig_pairs.iter().take(num_classes - 1).map(|p| p.1).sum();
        let total_variance = if total_variance > 1e-9 { total_variance } else { 1.0 };

        // 9. 판별 벡터 복원: w_i = L_inv^T * v_i
        // projection_matrix [target_dim][d]
        let mut projection_matrix = Vec::with_capacity(target_dim);
        let mut top_eigenvalues = Vec::with_capacity(target_dim);
        let mut explained_ratios = Vec::with_capacity(target_dim);

        for comp in 0..target_dim {
            let (orig_idx, eig_val) = eig_pairs[comp];
            let v_col = &v_mat[orig_idx]; // v_mat[col] 형태

            let mut w = vec![0.0f32; d];
            // w = L_inv^T * v => w[j] = sum_k L_inv[k][j] * v[k]
            for j in 0..d {
                let mut sum = 0.0f32;
                for k in j..d {
                    sum += l_inv[k][j] * v_col[k];
                }
                w[j] = sum;
            }

            // w 정규화: ||w|| = 1
            let norm = w.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for item in &mut w {
                *item /= norm;
            }

            projection_matrix.push(w);
            top_eigenvalues.push(eig_val);
            explained_ratios.push((eig_val / total_variance) * 100.0);
        }

        // 10. 4차원 투영 공간에서의 클래스별 중심점 사전 계산
        let mut projected_class_centroids = Vec::with_capacity(num_classes);
        for c in 0..num_classes {
            let mut diff = vec![0.0f32; d];
            for j in 0..d {
                diff[j] = class_means[c][j] - overall_mean[j];
            }
            let mut proj_c = vec![0.0f32; target_dim];
            for comp in 0..target_dim {
                let w = &projection_matrix[comp];
                let val: f32 = diff.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
                proj_c[comp] = val;
            }
            projected_class_centroids.push(proj_c);
        }

        Ok(Self {
            feature_dim: d,
            n_components: target_dim,
            num_classes,
            overall_mean,
            class_means,
            class_counts,
            projection_matrix,
            eigenvalues: top_eigenvalues,
            explained_variance_ratios: explained_ratios,
            projected_class_centroids,
        })
    }

    /// 단일 75차원 특징 벡터를 target_dim(예: 4차원)으로 투영합니다.
    pub fn transform_single(&self, x: &[f32]) -> Vec<f32> {
        let mut proj = vec![0.0f32; self.n_components];
        for comp in 0..self.n_components {
            let w = &self.projection_matrix[comp];
            let mut sum = 0.0f32;
            for j in 0..self.feature_dim.min(x.len()) {
                sum += (x[j] - self.overall_mean[j]) * w[j];
            }
            proj[comp] = sum;
        }
        proj
    }

    /// 여러 샘플을 일괄 투영합니다.
    pub fn transform_batch(&self, samples: &[Vec<f32>]) -> Vec<Vec<f32>> {
        samples.iter().map(|s| self.transform_single(s)).collect()
    }

    /// 투영 공간에서의 유클리드/마할라노비스 중심 거리를 기반으로 클래스를 예측합니다.
    pub fn predict(&self, x: &[f32]) -> usize {
        let z = self.transform_single(x);
        self.predict_projected(&z)
    }

    /// 이미 투영된 n_components 차원 벡터로부터 클래스를 예측합니다.
    pub fn predict_projected(&self, z: &[f32]) -> usize {
        let mut best_class = 0;
        let mut min_dist_sq = f32::MAX;

        for c in 0..self.num_classes {
            let centroid = &self.projected_class_centroids[c];
            let mut dist_sq = 0.0f32;
            for comp in 0..self.n_components {
                let diff = z[comp] - centroid[comp];
                dist_sq += diff * diff;
            }
            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                best_class = c;
            }
        }

        best_class
    }
}

// =========================================================================
// 선형대수 헬퍼 함수 (순수 Rust std 구현)
// =========================================================================

/// Cholesky 분해: A = L * L^T (L: 하삼각 행렬)
fn cholesky_decompose(a: &[Vec<f32>], d: usize) -> Option<Vec<Vec<f32>>> {
    let mut l = vec![vec![0.0f32; d]; d];

    for i in 0..d {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                let diag = l[j][j];
                if diag <= 1e-12 {
                    return None;
                }
                l[i][j] = sum / diag;
            }
        }
    }

    Some(l)
}

/// 하삼각 행렬 L의 역행렬 L_inv 계산 (L * L_inv = I)
fn invert_lower_triangular(l: &[Vec<f32>], d: usize) -> Vec<Vec<f32>> {
    let mut inv = vec![vec![0.0f32; d]; d];

    for i in 0..d {
        inv[i][i] = 1.0 / l[i][i].max(1e-12);
        for j in 0..i {
            let mut sum = 0.0f32;
            for k in j..i {
                sum += l[i][k] * inv[k][j];
            }
            inv[i][j] = -sum / l[i][i].max(1e-12);
        }
    }

    inv
}

/// Jacobi 고유치 알고리즘 (실수 대칭 행렬 M_mat)
/// 반환값: (고유값 벡터 [d], 고유벡터 열 행렬 [d][d])
fn jacobi_eigenvalue_sym(
    m_mat: &[Vec<f32>],
    d: usize,
    max_sweeps: usize,
    tol: f32,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut a = m_mat.to_vec();
    // V 초기화: 항등 행렬
    let mut v = vec![vec![0.0f32; d]; d];
    for i in 0..d {
        v[i][i] = 1.0;
    }

    for _sweep in 0..max_sweeps {
        let mut off_diag_sum = 0.0f32;
        for i in 0..d {
            for j in (i + 1)..d {
                off_diag_sum += a[i][j].abs();
            }
        }

        if off_diag_sum < tol {
            break;
        }

        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p][q];
                if apq.abs() < 1e-12 {
                    continue;
                }

                let app = a[p][p];
                let aqq = a[q][q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // A 업데이트
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                for r in 0..d {
                    if r != p && r != q {
                        let arp = a[r][p];
                        let arq = a[r][q];
                        a[r][p] = c * arp - s * arq;
                        a[p][r] = a[r][p];
                        a[r][q] = s * arp + c * arq;
                        a[q][r] = a[r][q];
                    }
                }

                // V 업데이트 (V = V * J)
                for r in 0..d {
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let mut eigenvalues = vec![0.0f32; d];
    for i in 0..d {
        eigenvalues[i] = a[i][i];
    }

    // v_cols[col][row] 형태로 전치하여 반환
    let mut eigenvectors_by_col = vec![vec![0.0f32; d]; d];
    for col in 0..d {
        for row in 0..d {
            eigenvectors_by_col[col][row] = v[row][col];
        }
    }

    (eigenvalues, eigenvectors_by_col)
}

// =========================================================================
// 데이터 정제 유틸리티: 3초 슬라이싱 & CSV 입출력
// =========================================================================

/// 원본 CSV에서 각 동작 블록의 안정적인 3초(150샘플 @ 50Hz)를 추출하여 반환합니다.
pub fn slice_stable_3s_dataset(
    csv_path: &Path,
    samples_per_block: usize, // 예: 150
) -> Result<(Vec<Vec<f32>>, Vec<usize>, Vec<String>, Vec<String>), String> {
    let file = File::open(csv_path)
        .map_err(|e| format!("CSV 파일을 열 수 없습니다: {} ({})", csv_path.display(), e))?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| "CSV 헤더가 비어 있습니다.".to_string())?
        .map_err(|e| e.to_string())?;

    let header_cols: Vec<String> = header_line.split(',').map(|s| s.trim().to_string()).collect();

    // 열 인덱스 탐색
    let label_idx = header_cols
        .iter()
        .position(|c| c == "label")
        .ok_or_else(|| "label 열을 찾을 수 없습니다.".to_string())?;
    let action_idx = header_cols
        .iter()
        .position(|c| c == "action_name")
        .ok_or_else(|| "action_name 열을 찾을 수 없습니다.".to_string())?;

    // 특징 열 탐색 (label과 action_name을 제외한 모든 열)
    let feat_indices: Vec<usize> = header_cols
        .iter()
        .enumerate()
        .filter(|(idx, col)| *idx != label_idx && *idx != action_idx && col.as_str() != "index")
        .map(|(idx, _)| idx)
        .collect();

    let _feat_names: Vec<String> = feat_indices
        .iter()
        .map(|&idx| header_cols[idx].clone())
        .collect();


    if feat_indices.is_empty() {
        return Err("특징 열이 존재하지 않습니다.".to_string());
    }

    // 전체 행 로드
    struct RawRow {
        features: Vec<f32>,
        label: usize,
        action_name: String,
    }

    let mut all_rows = Vec::new();
    for line_res in lines {
        let line = line_res.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if cols.len() <= label_idx || cols.len() <= action_idx {
            continue;
        }

        let label: usize = cols[label_idx].parse().unwrap_or(0);
        let action_name = cols[action_idx].to_string();

        let mut feats = Vec::with_capacity(feat_indices.len());
        for &idx in &feat_indices {
            let val: f32 = if idx < cols.len() {
                cols[idx].parse().unwrap_or(0.0)
            } else {
                0.0
            };
            feats.push(val);
        }

        all_rows.push(RawRow {
            features: feats,
            label,
            action_name,
        });
    }


    // 연속된 라벨별로 블록 분할
    let mut blocks: Vec<Vec<RawRow>> = Vec::new();
    let mut current_block: Vec<RawRow> = Vec::new();

    for row in all_rows {
        if let Some(last) = current_block.last() {
            if last.label != row.label {
                blocks.push(std::mem::take(&mut current_block));
            }
        }
        current_block.push(row);
    }
    if !current_block.is_empty() {
        blocks.push(current_block);
    }

    // 각 블록에서 중간 samples_per_block (150샘플) 추출
    let mut sliced_features = Vec::new();
    let mut sliced_labels = Vec::new();
    let mut sliced_actions = Vec::new();

    for block in blocks {
        let blen = block.len();
        if blen < samples_per_block {
            // 샘플 수가 모자라면 전체 사용
            for row in block {
                sliced_features.push(row.features);
                sliced_labels.push(row.label);
                sliced_actions.push(row.action_name);
            }
        } else {
            // 중간 슬라이스: start = (blen - samples_per_block) / 2
            let start = (blen - samples_per_block) / 2;
            for row in block.into_iter().skip(start).take(samples_per_block) {
                sliced_features.push(row.features);
                sliced_labels.push(row.label);
                sliced_actions.push(row.action_name);
            }
        }
    }

    Ok((sliced_features, sliced_labels, sliced_actions, header_cols))
}

/// 4차원으로 투영된 LDA 데이터셋을 CSV 파일로 저장합니다.
pub fn save_lda4d_csv(
    output_path: &Path,
    proj_data: &[Vec<f32>],
    labels: &[usize],
    action_names: &[String],
) -> Result<(), String> {
    let mut file = File::create(output_path)
        .map_err(|e| format!("출력 CSV 생성 실패: {} ({})", output_path.display(), e))?;

    writeln!(file, "index,ld1,ld2,ld3,ld4,label,action_name")
        .map_err(|e| e.to_string())?;

    for (i, row) in proj_data.iter().enumerate() {
        let ld1 = row.first().copied().unwrap_or(0.0);
        let ld2 = row.get(1).copied().unwrap_or(0.0);
        let ld3 = row.get(2).copied().unwrap_or(0.0);
        let ld4 = row.get(3).copied().unwrap_or(0.0);
        let label = labels.get(i).copied().unwrap_or(0);
        let action = action_names.get(i).cloned().unwrap_or_else(|| "Unknown".to_string());

        writeln!(
            file,
            "{},{:.6},{:.6},{:.6},{:.6},{},{}",
            i, ld1, ld2, ld3, ld4, label, action
        )
        .map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// 4D LDA 분류기 5-Fold 교차 검증
pub fn evaluate_lda_cross_validation(
    x: &[Vec<f32>],
    y: &[usize],
    num_classes: usize,
    n_components: usize,
) -> Result<crate::svm::ClassificationMetrics, String> {
    let n = x.len();
    let k_folds = 5;

    let mut class_indices = vec![Vec::new(); num_classes];
    for (i, &label) in y.iter().enumerate() {
        if label < num_classes {
            class_indices[label].push(i);
        }
    }

    let mut fold_test_indices = vec![Vec::new(); k_folds];
    for list in &class_indices {
        for (i, &idx) in list.iter().enumerate() {
            let fold = i % k_folds;
            fold_test_indices[fold].push(idx);
        }
    }

    let mut all_y_true = Vec::with_capacity(n);
    let mut all_y_pred = Vec::with_capacity(n);

    for fold in 0..k_folds {
        let test_set = &fold_test_indices[fold];
        let mut is_test = vec![false; n];
        for &idx in test_set {
            is_test[idx] = true;
        }

        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        let mut test_x = Vec::new();
        let mut test_y = Vec::new();

        for i in 0..n {
            if is_test[i] {
                test_x.push(x[i].clone());
                test_y.push(y[i]);
            } else {
                train_x.push(x[i].clone());
                train_y.push(y[i]);
            }
        }

        let model = LdaModel::fit(&train_x, &train_y, n_components)?;
        let preds: Vec<usize> = test_x.iter().map(|sample| model.predict(sample)).collect();

        all_y_true.extend(test_y);
        all_y_pred.extend(preds);
    }

    Ok(crate::svm::ClassificationMetrics::compute(
        &all_y_true,
        &all_y_pred,
        num_classes,
    ))
}

/// 대화형 3초 4D LDA & SVM 분석 및 학습 파이프라인
pub fn run_lda_svm_analysis_interactive() -> Result<(), Box<dyn std::error::Error>> {
    use crate::svm::{cross_validate_svm, MultiClassSvm, SvmConfig, SvmKernel};
    use glob::glob;
    use std::io::{self, Write};
    use std::path::PathBuf;
    use std::time::Instant;

    println!("\n========================================================");
    println!(" 🔬 EMG 3초 정제 4차원 LDA & SVM 분류 파이프라인 (Pure Rust)");
    println!("========================================================");

    // 1. 손가락 데이터셋 파일 탐색
    let mut candidate_files: Vec<PathBuf> = Vec::new();
    for entry in glob("emg_dataset_fingers*.csv")? {
        if let Ok(path) = entry {
            if !path.to_string_lossy().contains("clean") && !path.to_string_lossy().contains("lda4d") {
                candidate_files.push(path);
            }
        }
    }
    candidate_files.sort_by(|a, b| b.cmp(a)); // 최신순

    if candidate_files.is_empty() {
        for entry in glob("emg_dataset_*.csv")? {
            if let Ok(path) = entry {
                if !path.to_string_lossy().contains("clean") && !path.to_string_lossy().contains("lda4d") {
                    candidate_files.push(path);
                }
            }
        }
        candidate_files.sort_by(|a, b| b.cmp(a));
    }

    if candidate_files.is_empty() {
        eprintln!("❌ 처리할 EMG CSV 데이터셋 파일을 찾을 수 없습니다.");
        return Ok(());
    }

    println!("\n📁 발견된 데이터셋 목록:");
    for (i, p) in candidate_files.iter().enumerate() {
        println!(" [{}] {}", i + 1, p.display());
    }
    print!("분석할 데이터셋 번호를 선택하세요 (1-{}, 기본 1) > ", candidate_files.len());
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice: usize = input.trim().parse().unwrap_or(1);
    let selected_csv = candidate_files
        .get(choice.saturating_sub(1))
        .unwrap_or(&candidate_files[0]);

    println!("\n>> 선택된 파일: {}", selected_csv.display());

    // 2. 3초(150샘플 @ 50Hz) 정밀 슬라이싱 수행
    println!(">> 각 제스처 구간에서 전이 오차를 제외한 중앙 3초(150 샘플) 정제 추출 중...");
    let (clean_x, clean_y, clean_actions, header_cols) =
        slice_stable_3s_dataset(selected_csv, 150)
            .map_err(|e| format!("데이터 정제 실패: {}", e))?;

    let n = clean_x.len();
    let d = clean_x[0].len();
    let num_classes = clean_y.iter().copied().max().unwrap_or(0) + 1;

    // 고유 클래스 이름 추출
    let mut class_names = vec![String::new(); num_classes];
    for (label, act) in clean_y.iter().zip(clean_actions.iter()) {
        if class_names[*label].is_empty() {
            class_names[*label] = act.clone();
        }
    }
    for (c, name) in class_names.iter_mut().enumerate() {
        if name.is_empty() {
            *name = format!("Class {}", c);
        }
    }

    println!("✅ 정제 완료: 총 {} 개 샘플 (특징 차원: {}D, 클래스 수: {}개)", n, d, num_classes);
    for (c, name) in class_names.iter().enumerate() {
        let count = clean_y.iter().filter(|&&y| y == c).count();
        println!("   - C{}: {:<22} [{} 샘플]", c, name, count);
    }

    // 정제된 3초 데이터셋 CSV로 저장
    let clean_csv_path = Path::new("emg_dataset_fingers_3s_clean.csv");
    {
        let mut f = File::create(clean_csv_path)?;
        writeln!(f, "{}", header_cols.join(","))?;
        for (i, feats) in clean_x.iter().enumerate() {
            let feat_strs: Vec<String> = feats.iter().map(|v| format!("{:.6}", v)).collect();
            writeln!(
                f,
                "{},{},{}",
                feat_strs.join(","),
                clean_y[i],
                clean_actions[i]
            )?;
        }
    }

    println!("💾 3초 정제 데이터셋 저장: {}", clean_csv_path.display());

    // 3. 순수 Rust 4D LDA 학습 및 축소 투영
    println!("\n========================================================");
    println!(" 📐 [Step 1] 순수 Rust 4차원 선형 판별 분석 (LDA) 학습");
    println!("========================================================");
    let lda_start = Instant::now();
    let target_lda_dim = 4.min(num_classes - 1);
    let lda_model = LdaModel::fit(&clean_x, &clean_y, target_lda_dim)
        .map_err(|e| format!("LDA 학습 실패: {}", e))?;
    let lda_duration = lda_start.elapsed();

    println!("⚡ LDA 학습 완료 ({:?} 소요)", lda_duration);
    println!("📊 판별 축(LD1 ~ LD{})별 고유값 및 분별 분산 기여율:", target_lda_dim);
    let mut cumulative_ratio = 0.0f32;
    for (comp, (eig, ratio)) in lda_model
        .eigenvalues
        .iter()
        .zip(lda_model.explained_variance_ratios.iter())
        .enumerate()
    {
        cumulative_ratio += ratio;
        println!(
            "   - LD{}: 고유값 = {:>10.4} | 기여율 = {:>6.2}% (누적 = {:>6.2}%)",
            comp + 1,
            eig,
            ratio,
            cumulative_ratio
        );
    }

    // 4차원 투영 데이터 생성
    let lda4d_x = lda_model.transform_batch(&clean_x);
    let lda4d_csv_path = Path::new("emg_dataset_fingers_3s_lda4d.csv");
    save_lda4d_csv(lda4d_csv_path, &lda4d_x, &clean_y, &clean_actions)
        .map_err(|e| format!("4D LDA CSV 저장 실패: {}", e))?;
    println!("💾 4차원 투영 데이터셋 저장 완료: {}", lda4d_csv_path.display());

    // 4. 모델별 성능 비교 실험 (5-Fold Stratified Cross Validation)
    println!("\n========================================================");
    println!(" 🧪 [Step 2] 5-Fold 교차 검증(Cross-Validation) 모델 비교 평가");
    println!("========================================================");
    println!(" * 공정한 평가를 위해 5-Fold Stratified 교차 검증으로 일반화 정확도를 측정합니다.\n");

    // 4-A. 4D LDA 기반 분류기 평가
    println!("🔹 [Model 1] 4D LDA 판별 분류기 (Centroid / Gaussian Bayes)");
    let lda_metrics = evaluate_lda_cross_validation(&clean_x, &clean_y, num_classes, target_lda_dim)?;
    lda_metrics.print_report(&class_names);

    // 4-B. 4D LDA 공간 + Linear SVM (Dual Coordinate Descent)
    println!("\n🔹 [Model 2] 4D LDA 투영 공간 + Linear SVM (One-vs-Rest, C=1.0)");
    let svm_linear_config = SvmConfig {
        kernel: SvmKernel::Linear,
        c: 1.0,
        max_iter: 100,
        tol: 1e-4,
    };
    let lda_linear_svm_metrics = cross_validate_svm(
        &lda4d_x,
        &clean_y,
        num_classes,
        svm_linear_config.clone(),
        5,
        &class_names,
    )?;
    lda_linear_svm_metrics.print_report(&class_names);

    // 4-C. 4D LDA 공간 + RBF Kernel SVM (Non-linear Gaussian Kernel)
    println!("\n🔹 [Model 3] 4D LDA 투영 공간 + RBF Kernel SVM (Gamma=0.25, C=2.0)");
    let svm_rbf_config = SvmConfig {
        kernel: SvmKernel::Rbf { gamma: 0.25 },
        c: 2.0,
        max_iter: 100,
        tol: 1e-4,
    };
    let lda_rbf_svm_metrics = cross_validate_svm(
        &lda4d_x,
        &clean_y,
        num_classes,
        svm_rbf_config.clone(),
        5,
        &class_names,
    )?;
    lda_rbf_svm_metrics.print_report(&class_names);

    // 4-D. 75D 원본 특징 + Linear SVM
    println!("\n🔹 [Model 4] 75차원 원본 특징 + Linear SVM (One-vs-Rest, C=0.5)");
    let raw_svm_linear_config = SvmConfig {
        kernel: SvmKernel::Linear,
        c: 0.5,
        max_iter: 80,
        tol: 1e-4,
    };
    let raw_linear_svm_metrics = cross_validate_svm(
        &clean_x,
        &clean_y,
        num_classes,
        raw_svm_linear_config,
        5,
        &class_names,
    )?;
    raw_linear_svm_metrics.print_report(&class_names);

    // 5. 최종 종합 요약 테이블
    println!("\n========================================================");
    println!(" 🏆 최종 분류기 5-Fold 교차 검증 성능 종합 요약");
    println!("========================================================");
    println!(
        "{:<35} | {:>10} | {:>12}",
        "모델 구조", "교차검증 정확도", "Macro F1"
    );
    println!("------------------------------------------------------------------");

    let models_summary = [
        ("1. 4D LDA Centroid Classifier", &lda_metrics),
        ("2. 4D LDA + Linear SVM (OvR)", &lda_linear_svm_metrics),
        ("3. 4D LDA + RBF Kernel SVM (OvR)", &lda_rbf_svm_metrics),
        ("4. 75D Raw + Linear SVM (OvR)", &raw_linear_svm_metrics),
    ];

    for (name, m) in &models_summary {
        let avg_f1: f32 = m.f1_scores.iter().sum::<f32>() / num_classes as f32;
        println!(
            "{:<35} | {:>9.2}% | {:>11.2}%",
            name, m.accuracy, avg_f1
        );
    }
    println!("========================================================");

    // 6. 최적의 모델 전체 데이터셋으로 학습 후 파일 저장
    println!("\n>> 전체 3초 정제 데이터셋(2,700 샘플)으로 최종 최적 SVM 모델 학습 중...");
    let final_svm = MultiClassSvm::train(
        &lda4d_x,
        &clean_y,
        num_classes,
        svm_rbf_config,
        Some(class_names.clone()),
    )?;

    let lda_model_path = Path::new("emg_lda4d_model.json");
    let svm_model_path = Path::new("emg_svm_model.json");

    let lda_json = serde_json::to_string_pretty(&lda_model)?;
    std::fs::write(lda_model_path, lda_json)?;

    let svm_json = serde_json::to_string_pretty(&final_svm)?;
    std::fs::write(svm_model_path, svm_json)?;

    println!("💾 최종 LDA 모델 저장 완료: {}", lda_model_path.display());
    println!("💾 최종 SVM 모델 저장 완료: {}", svm_model_path.display());
    println!("\n🎉 모든 분석 및 모델 학습이 성공적으로 완료되었습니다!");

    Ok(())
}

/// 4D LDA & SVM 모델을 이용한 실시간 추론 파이프라인
pub fn run_lda_svm_realtime_inference() -> Result<(), Box<dyn std::error::Error>> {
    use crate::fft::{EmgFftProcessor, NUM_CHANNELS, TOTAL_FEATURES_ALL};
    use crate::svm::MultiClassSvm;
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;
    use std::time::Duration;

    let lda_path = Path::new("emg_lda4d_model.json");
    let svm_path = Path::new("emg_svm_model.json");

    if !lda_path.exists() || !svm_path.exists() {
        println!("\n❌ [에러] LDA 모델 또는 SVM 모델 파일이 없습니다.");
        println!("먼저 메뉴 [7]번(3초 정제 4D LDA & SVM 분석 및 학습)을 실행하여 모델을 생성해주세요.");
        return Ok(());
    }

    println!("\n[알림] 저장된 4D LDA 및 SVM 모델 로드 중...");
    let lda_str = std::fs::read_to_string(lda_path)?;
    let lda_model: LdaModel = serde_json::from_str(&lda_str)?;

    let svm_str = std::fs::read_to_string(svm_path)?;
    let svm_model: MultiClassSvm = serde_json::from_str(&svm_str)?;

    println!("✔ 4D LDA 모델 로드 완료 (입력: {}D -> 판별 공간: {}D)", lda_model.feature_dim, lda_model.n_components);
    println!("✔ SVM 모델 로드 완료 (클래스: {}개 - {})", svm_model.num_classes, svm_model.class_names.join(", "));

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
            return Ok(());
        }
    };

    let mut port_write = port.try_clone().ok();
    if port_write.is_some() {
        println!("✔ 아두이노 양방향 시리얼 통신 활성화 (0:정적휴식/잡음, 1:진짜주먹 역전송)");
    } else {
        println!("⚠️ 시리얼 포트 복제 실패: 단방향(수신 전용)으로 실행됩니다.");
    }

    let mut reader = BufReader::new(port);

    // 1.5초 휴식기 영점 보정
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
                    std::io::stdout().flush().unwrap();
                }
            }
        }
    }

    if !baseline_samples.is_empty() {
        let mut rest_baseline = [0.0f32; NUM_CHANNELS];
        for s in &baseline_samples {
            for ch in 0..NUM_CHANNELS {
                rest_baseline[ch] += s[ch];
            }
        }
        for ch in 0..NUM_CHANNELS {
            rest_baseline[ch] /= baseline_samples.len() as f32;
        }
        println!(
            "\n✔ [영점 보정 완료] 착용 기준선: [Ch0: {:.0}, Ch1: {:.0}, Ch2: {:.0}, Ch3: {:.0}, Ch4: {:.0}]",
            rest_baseline[0], rest_baseline[1], rest_baseline[2], rest_baseline[3], rest_baseline[4]
        );
    }

    let mut processor = EmgFftProcessor::new();
    let num_classes = svm_model.num_classes;
    let mut ema_probs = vec![0.0f32; num_classes];
    ema_probs[0] = 1.0;
    let alpha = 0.20f32;
    let mut last_sent_val: Option<u8> = None;
    let mut frame_count: u64 = 0;

    println!("\n==========================================================");
    println!(" 🚀 [4D LDA + SVM] 실시간 손가락 추론 시작! (종료: Ctrl + C)");
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

                let current_features: Vec<f32>;
                let total_ac_rms: f32;

                if parts.len() >= 75 {
                    current_features = parts[..75].to_vec();
                    let feats_arr: &[f32; TOTAL_FEATURES_ALL] = current_features.as_slice().try_into().unwrap();
                    total_ac_rms = EmgFftProcessor::extract_total_ac_rms(feats_arr);
                } else if parts.len() == NUM_CHANNELS {
                    let mut raw_arr = [0.0f32; NUM_CHANNELS];
                    raw_arr.copy_from_slice(&parts[..NUM_CHANNELS]);
                    current_features = processor.process_sample_all(&raw_arr).to_vec();
                    let feats_arr: &[f32; TOTAL_FEATURES_ALL] = current_features.as_slice().try_into().unwrap();
                    total_ac_rms = EmgFftProcessor::extract_total_ac_rms(feats_arr);
                } else {
                    continue;
                }

                // 1. 75차원 특징 -> 4차원 LDA 투영
                let z = lda_model.transform_single(&current_features);

                // 2. SVM 결정 점수 계산
                let scores = svm_model.decision_scores(&z);
                let min_score = scores.iter().copied().fold(f32::INFINITY, f32::min);
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

                // 3. 지능형 확률 산출 (RBF 커널 유효성 검사 및 4D LDA Centroid 자동 연동)
                // 만약 신호가 너무 크거나 외곽으로 벗어나 RBF 점수가 모두 0으로 언더플로우되면,
                // 4D LDA 판별 중심 거리 기반 가우시안 사후 확률로 즉시 전환하여 16.7% 멈춤 현상 원천 방지!
                let raw_probs: Vec<f32> = if (max_score - min_score).abs() > 1e-3 {
                    let exp_scores: Vec<f32> = scores.iter().map(|s| ((s - max_score) * 2.0).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum::<f32>().max(1e-12);
                    exp_scores.iter().map(|e| e / sum_exp).collect()
                } else {
                    // 4D LDA Centroid 거리 기반 가우시안 확률 계산
                    let mut dists = Vec::with_capacity(num_classes);
                    for c in 0..num_classes {
                        let centroid = &lda_model.projected_class_centroids[c];
                        let mut d_sq = 0.0f32;
                        for comp in 0..lda_model.n_components {
                            let diff = z[comp] - centroid[comp];
                            d_sq += diff * diff;
                        }
                        dists.push(d_sq);
                    }
                    let min_d = dists.iter().copied().fold(f32::INFINITY, f32::min);
                    let temp = (min_d * 0.15).max(15.0);
                    let exp_d: Vec<f32> = dists.iter().map(|d| (-(d - min_d) / temp).exp()).collect();
                    let sum_d: f32 = exp_d.iter().sum::<f32>().max(1e-12);
                    exp_d.iter().map(|e| e / sum_d).collect()
                };

                // 4. 반응성 높은 EMA 스무딩 (alpha = 0.35)
                for i in 0..num_classes {
                    ema_probs[i] = alpha * raw_probs[i] + (1.0 - alpha) * ema_probs[i];
                }

                // 센서 전극 들뜸(포화) 감지
                let mut sat_ch: Option<usize> = None;
                if parts.len() == NUM_CHANNELS {
                    for (ch, &p) in parts.iter().enumerate() {
                        if p >= 1020.0 || p <= 3.0 {
                            sat_ch = Some(ch);
                            break;
                        }
                    }
                }

                let mut best_idx = 0;
                let mut best_prob = 0.0f32;
                for (i, &p) in ema_probs.iter().enumerate() {
                    if p > best_prob {
                        best_prob = p;
                        best_idx = i;
                    }
                }

                let class_name = svm_model.class_names.get(best_idx).map(|s| s.as_str()).unwrap_or("알 수 없음");
                // 🚀 [아두이노 역전송 판별] 진짜 주먹 / 손 쥐기인 경우 1, 그 외는 0
                let is_fist = class_name.contains("주먹") || class_name.contains("손 쥐기") || class_name.to_lowercase().contains("fist");
                let out_val: u8 = if is_fist { 1 } else { 0 };

                // 상태 변경 시 즉시 전송 + 25프레임(~0.5초) 주기적 동기화 하트비트 전송
                frame_count = frame_count.wrapping_add(1);
                let should_send = match last_sent_val {
                    Some(last) => last != out_val || frame_count % 25 == 0,
                    None => true,
                };

                if should_send {
                    if let Some(ref mut writer) = port_write {
                        let msg = format!("{}\n", out_val);
                        if writer.write_all(msg.as_bytes()).is_ok() {
                            let _ = writer.flush();
                            last_sent_val = Some(out_val);
                        }
                    }
                }

                let tx_tag = if out_val == 1 { "📤아두이노: 1 (주먹)" } else { "📤아두이노: 0 (휴식)" };

                let prob_str = svm_model
                    .class_names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let p = ema_probs.get(i).copied().unwrap_or(0.0) * 100.0;
                        let short_name = name.split_whitespace().next().unwrap_or(name);
                        format!("{}:{:4.1}%", short_name, p)
                    })
                    .collect::<Vec<_>>()
                    .join(" | ");

                let ld_str = format!(
                    "[{:5.1},{:5.1},{:5.1},{:5.1}]",
                    z.get(0).copied().unwrap_or(0.0),
                    z.get(1).copied().unwrap_or(0.0),
                    z.get(2).copied().unwrap_or(0.0),
                    z.get(3).copied().unwrap_or(0.0)
                );

                let warn_tag = if let Some(ch) = sat_ch {
                    format!("⚠️Ch{}포화!", ch)
                } else {
                    format!("RMS:{:4.1}", total_ac_rms)
                };

                print!(
                    "\r🤖 [4D LDA] {:<18} | [{}] | 4D:{} | {} | {}  ",
                    class_name,
                    prob_str,
                    ld_str,
                    warn_tag,
                    tx_tag,
                );
                std::io::stdout().flush().unwrap();

            }
            Err(_) => continue,
        }
    }
}


