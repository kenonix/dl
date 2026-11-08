//! Support Vector Machine (SVM) 모듈
//!
//! 고차원(75차원) 원본 EMG 특징 또는 4차원 LDA 투영 공간에서
//! 다중 클래스(휴식 및 다섯 손가락 등)를 고정밀로 분류하는
//! One-vs-Rest (OvR) 구조의 순수 Rust SVM 구현체입니다.
//!
//! - Linear SVM: Dual Coordinate Descent (DCD, Liblinear 방식) 기반 고속 선형 분류
//! - RBF Kernel SVM: 가우시안 비선형 커널을 통한 유연한 결정 경계 형성
//! - 5-Fold 교차 검증, 혼동 행렬(Confusion Matrix), 클래스별 F1-Score 자동 산출

use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// SVM 커널 타입
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SvmKernel {
    Linear,
    Rbf { gamma: f32 },
}

/// SVM 학습 하이퍼파라미터 설정
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvmConfig {
    pub kernel: SvmKernel,
    pub c: f32,
    pub max_iter: usize,
    pub tol: f32,
}

impl Default for SvmConfig {
    fn default() -> Self {
        Self {
            kernel: SvmKernel::Linear,
            c: 1.0,
            max_iter: 100,
            tol: 1e-4,
        }
    }
}

/// 이진 선형 SVM 모델
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryLinearSvm {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl BinaryLinearSvm {
    /// Dual Coordinate Descent로 L1-loss SVM 학습
    pub fn train(x: &[Vec<f32>], y: &[f32], c: f32, max_iter: usize, tol: f32) -> Self {
        let n = x.len();
        if n == 0 {
            return Self {
                weights: Vec::new(),
                bias: 0.0,
            };
        }
        let d = x[0].len();

        // 편향(bias)을 모델링하기 위해 특징 벡터 끝에 1.0 추가
        let mut q_diag = vec![0.0f32; n];
        for i in 0..n {
            let mut sum_sq = 1.0f32; // bias^2 = 1.0
            for j in 0..d {
                sum_sq += x[i][j] * x[i][j];
            }
            q_diag[i] = sum_sq.max(1e-12);
        }

        let mut alpha = vec![0.0f32; n];
        let mut w = vec![0.0f32; d];
        let mut bias = 0.0f32;

        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();

        for _iter in 0..max_iter {
            indices.shuffle(&mut rng);
            let mut max_pg = 0.0f32;

            for &i in &indices {
                let yi = y[i];
                let mut dot = bias;
                for j in 0..d {
                    dot += w[j] * x[i][j];
                }

                let g = yi * dot - 1.0;
                let old_alpha = alpha[i];

                // 투영 그래디언트
                let pg = if old_alpha <= 0.0 {
                    g.min(0.0)
                } else if old_alpha >= c {
                    g.max(0.0)
                } else {
                    g
                };

                max_pg = max_pg.max(pg.abs());

                if pg.abs() > tol {
                    let new_alpha = (old_alpha - g / q_diag[i]).clamp(0.0, c);
                    let delta = new_alpha - old_alpha;
                    if delta.abs() > 1e-12 {
                        let coeff = delta * yi;
                        for j in 0..d {
                            w[j] += coeff * x[i][j];
                        }
                        bias += coeff * 1.0;
                        alpha[i] = new_alpha;
                    }
                }
            }

            if max_pg < tol {
                break;
            }
        }

        Self { weights: w, bias }
    }

    /// 마진 결정 값 계산: f(x) = w^T x + b
    pub fn decision_function(&self, x: &[f32]) -> f32 {
        let mut sum = self.bias;
        let d = self.weights.len().min(x.len());
        for j in 0..d {
            sum += self.weights[j] * x[j];
        }
        sum
    }
}

/// 이진 RBF 커널 SVM 모델
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryRbfSvm {
    pub support_vectors: Vec<Vec<f32>>,
    pub sv_alpha_y: Vec<f32>, // alpha_i * y_i
    pub bias: f32,
    pub gamma: f32,
}

impl BinaryRbfSvm {
    /// Kernel Dual Coordinate Descent 학습
    pub fn train(x: &[Vec<f32>], y: &[f32], c: f32, gamma: f32, max_iter: usize, tol: f32) -> Self {
        let n = x.len();
        if n == 0 {
            return Self {
                support_vectors: Vec::new(),
                sv_alpha_y: Vec::new(),
                bias: 0.0,
                gamma,
            };
        }
        let d = x[0].len();

        let mut alpha = vec![0.0f32; n];
        let mut g = vec![-1.0f32; n]; // g_i = y_i * f(x_i) - 1

        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();

        // RBF 커널 계산 클로저
        let rbf = |xi: &[f32], xj: &[f32]| -> f32 {
            let mut dist_sq = 0.0f32;
            for k in 0..d {
                let diff = xi[k] - xj[k];
                dist_sq += diff * diff;
            }
            (-gamma * dist_sq).exp()
        };

        for _iter in 0..max_iter {
            indices.shuffle(&mut rng);
            let mut max_pg = 0.0f32;

            for &i in &indices {
                let gi = g[i];
                let old_alpha = alpha[i];

                let pg = if old_alpha <= 0.0 {
                    gi.min(0.0)
                } else if old_alpha >= c {
                    gi.max(0.0)
                } else {
                    gi
                };

                max_pg = max_pg.max(pg.abs());

                if pg.abs() > tol {
                    let k_ii = 1.0f32; // exp(0) = 1.0
                    let new_alpha = (old_alpha - gi / k_ii).clamp(0.0, c);
                    let delta = new_alpha - old_alpha;

                    if delta.abs() > 1e-12 {
                        let delta_ay = delta * y[i];
                        alpha[i] = new_alpha;

                        // 그래디언트 업데이트: g_j += y_j * delta_ay * K(x_i, x_j)
                        let xi = &x[i];
                        for j in 0..n {
                            let k_ij = rbf(xi, &x[j]);
                            g[j] += y[j] * delta_ay * k_ij;
                        }
                    }
                }
            }

            if max_pg < tol {
                break;
            }
        }

        // 서포트 벡터 추출 (alpha_i > 1e-5)
        let mut svs = Vec::new();
        let mut sv_ay = Vec::new();
        let mut bias_sum = 0.0f32;
        let mut bias_count = 0usize;

        for i in 0..n {
            if alpha[i] > 1e-5 {
                svs.push(x[i].clone());
                sv_ay.push(alpha[i] * y[i]);

                // 경계 내 서포트 벡터 (0 < alpha < C)를 이용해 편향 b 추정
                if alpha[i] < c - 1e-5 {
                    // y_i * (f(x_i) - b) + b * y_i = 1 => b = y_i - sum_j alpha_j y_j K(x_i, x_j)
                    let pred_no_b: f32 = (0..n)
                        .filter(|&k| alpha[k] > 1e-5)
                        .map(|k| alpha[k] * y[k] * rbf(&x[k], &x[i]))
                        .sum();
                    bias_sum += y[i] - pred_no_b;
                    bias_count += 1;
                }
            }
        }

        let bias = if bias_count > 0 {
            bias_sum / bias_count as f32
        } else {
            0.0
        };

        Self {
            support_vectors: svs,
            sv_alpha_y: sv_ay,
            bias,
            gamma,
        }
    }

    /// 마진 결정 값 계산: f(x) = sum_i alpha_i y_i K(x_i, x) + b
    pub fn decision_function(&self, x: &[f32]) -> f32 {
        let mut sum = self.bias;
        let d = x.len();
        for (sv, &ay) in self.support_vectors.iter().zip(self.sv_alpha_y.iter()) {
            let mut dist_sq = 0.0f32;
            for j in 0..d {
                let diff = sv[j] - x[j];
                dist_sq += diff * diff;
            }
            sum += ay * (-self.gamma * dist_sq).exp();
        }
        sum
    }
}

/// 이진 분류기 열거형
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinaryClassifier {
    Linear(BinaryLinearSvm),
    Rbf(BinaryRbfSvm),
}

impl BinaryClassifier {
    pub fn decision_function(&self, x: &[f32]) -> f32 {
        match self {
            BinaryClassifier::Linear(svm) => svm.decision_function(x),
            BinaryClassifier::Rbf(svm) => svm.decision_function(x),
        }
    }
}

/// 다중 클래스 One-vs-Rest (OvR) SVM 분류기
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiClassSvm {
    pub num_classes: usize,
    pub feature_dim: usize,
    pub config: SvmConfig,
    pub classifiers: Vec<BinaryClassifier>,
    pub class_names: Vec<String>,
}

impl MultiClassSvm {
    /// X [N, D] 와 y [N] (0 .. num_classes-1) 로부터 One-vs-Rest SVM 학습
    pub fn train(
        x: &[Vec<f32>],
        y: &[usize],
        num_classes: usize,
        config: SvmConfig,
        class_names: Option<Vec<String>>,
    ) -> Result<Self, String> {
        let n = x.len();
        if n == 0 {
            return Err("훈련 데이터가 비어 있습니다.".to_string());
        }
        let d = x[0].len();

        let mut classifiers = Vec::with_capacity(num_classes);

        for target_c in 0..num_classes {
            // 이진 레이블 생성 (+1: target_c, -1: 나머지)
            let mut y_bin = vec![-1.0f32; n];
            let mut pos_count = 0usize;
            for i in 0..n {
                if y[i] == target_c {
                    y_bin[i] = 1.0;
                    pos_count += 1;
                }
            }

            if pos_count == 0 {
                return Err(format!("클래스 {}의 샘플이 존재하지 않습니다.", target_c));
            }

            let clf = match config.kernel {
                SvmKernel::Linear => {
                    let bin_svm =
                        BinaryLinearSvm::train(x, &y_bin, config.c, config.max_iter, config.tol);
                    BinaryClassifier::Linear(bin_svm)
                }
                SvmKernel::Rbf { gamma } => {
                    let bin_svm = BinaryRbfSvm::train(
                        x,
                        &y_bin,
                        config.c,
                        gamma,
                        config.max_iter,
                        config.tol,
                    );
                    BinaryClassifier::Rbf(bin_svm)
                }
            };
            classifiers.push(clf);
        }

        let names = class_names.unwrap_or_else(|| {
            (0..num_classes).map(|c| format!("Class {}", c)).collect()
        });

        Ok(Self {
            num_classes,
            feature_dim: d,
            config,
            classifiers,
            class_names: names,
        })
    }

    /// 단일 샘플의 마진 점수 벡터 반환
    pub fn decision_scores(&self, x: &[f32]) -> Vec<f32> {
        self.classifiers.iter().map(|clf| clf.decision_function(x)).collect()
    }

    /// 가장 높은 마진 점수를 갖는 클래스 예측 및 Softmax 확률 반환
    pub fn predict(&self, x: &[f32]) -> (usize, Vec<f32>) {
        let scores = self.decision_scores(x);
        let mut best_class = 0;
        let mut max_score = f32::NEG_INFINITY;

        for (c, &s) in scores.iter().enumerate() {
            if s > max_score {
                max_score = s;
                best_class = c;
            }
        }

        // Softmax 확률 산출
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum::<f32>().max(1e-12);
        let probs: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        (best_class, probs)
    }

    /// 일괄 예측
    pub fn predict_batch(&self, samples: &[Vec<f32>]) -> Vec<usize> {
        samples.iter().map(|s| self.predict(s).0).collect()
    }
}

// =========================================================================
// 모델 성능 평가 지표 (Confusion Matrix & Classification Report)
// =========================================================================

/// 분류 성능 평가지표
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    pub accuracy: f32,
    pub confusion_matrix: Vec<Vec<usize>>, // [true_label][pred_label]
    pub precisions: Vec<f32>,
    pub recalls: Vec<f32>,
    pub f1_scores: Vec<f32>,
    pub class_counts: Vec<usize>,
}

impl ClassificationMetrics {
    pub fn compute(y_true: &[usize], y_pred: &[usize], num_classes: usize) -> Self {
        let n = y_true.len();
        let mut cm = vec![vec![0usize; num_classes]; num_classes];
        let mut correct = 0usize;

        for i in 0..n {
            let t = y_true[i];
            let p = y_pred[i];
            if t < num_classes && p < num_classes {
                cm[t][p] += 1;
                if t == p {
                    correct += 1;
                }
            }
        }

        let accuracy = if n > 0 {
            (correct as f32 / n as f32) * 100.0
        } else {
            0.0
        };

        let mut precisions = vec![0.0f32; num_classes];
        let mut recalls = vec![0.0f32; num_classes];
        let mut f1_scores = vec![0.0f32; num_classes];
        let mut class_counts = vec![0usize; num_classes];

        for c in 0..num_classes {
            let tp = cm[c][c] as f32;
            let actual_total: usize = cm[c].iter().sum();
            class_counts[c] = actual_total;

            let pred_total: usize = (0..num_classes).map(|r| cm[r][c]).sum();

            let prec = if pred_total > 0 {
                tp / pred_total as f32
            } else {
                0.0
            };
            let rec = if actual_total > 0 {
                tp / actual_total as f32
            } else {
                0.0
            };
            let f1 = if prec + rec > 1e-6 {
                2.0 * prec * rec / (prec + rec)
            } else {
                0.0
            };

            precisions[c] = prec * 100.0;
            recalls[c] = rec * 100.0;
            f1_scores[c] = f1 * 100.0;
        }

        Self {
            accuracy,
            confusion_matrix: cm,
            precisions,
            recalls,
            f1_scores,
            class_counts,
        }
    }

    /// 보기 좋은 콘솔 테이블로 출력
    pub fn print_report(&self, class_names: &[String]) {
        let num_classes = self.confusion_matrix.len();
        println!("--------------------------------------------------------------------------");
        println!(
            "{:<16} | {:>9} | {:>9} | {:>9} | {:>8}",
            "Class Name", "Precision", "Recall", "F1-Score", "Support"
        );
        println!("--------------------------------------------------------------------------");

        for c in 0..num_classes {
            let name = class_names
                .get(c)
                .cloned()
                .unwrap_or_else(|| format!("Class {}", c));
            println!(
                "{:<16} | {:>8.2}% | {:>8.2}% | {:>8.2}% | {:>8}",
                name, self.precisions[c], self.recalls[c], self.f1_scores[c], self.class_counts[c]
            );
        }

        println!("--------------------------------------------------------------------------");
        let avg_f1: f32 = self.f1_scores.iter().sum::<f32>() / num_classes as f32;
        println!(
            "총 정확도 (Accuracy): {:>6.2}%   (Macro F1-Score: {:>6.2}%)",
            self.accuracy, avg_f1
        );
        println!("--------------------------------------------------------------------------");

        println!("\n📊 혼동 행렬 (Confusion Matrix: 행=실제 정답, 열=모델 예측):");
        print!("          ");
        for c in 0..num_classes {
            print!("C{:1}   ", c);
        }
        println!();
        for r in 0..num_classes {
            let short_name = class_names
                .get(r)
                .map(|s| s.chars().take(8).collect::<String>())
                .unwrap_or_else(|| format!("C{}", r));
            print!("{:>8} [", short_name);
            for c in 0..num_classes {
                print!("{:4} ", self.confusion_matrix[r][c]);
            }
            println!("]");
        }
        println!();
    }
}

// =========================================================================
// 5-Fold Stratified Cross Validation
// =========================================================================

/// 5-Fold Stratified 교차 검증을 수행하여 일반화 성능을 정확히 측정합니다.
pub fn cross_validate_svm(
    x: &[Vec<f32>],
    y: &[usize],
    num_classes: usize,
    config: SvmConfig,
    k_folds: usize,
    class_names: &[String],
) -> Result<ClassificationMetrics, String> {
    let n = x.len();
    if n < k_folds * num_classes {
        return Err("교차 검증을 위한 샘플 수가 부족합니다.".to_string());
    }

    // 클래스별 인덱스 분류
    let mut class_indices = vec![Vec::new(); num_classes];
    for (i, &label) in y.iter().enumerate() {
        if label < num_classes {
            class_indices[label].push(i);
        }
    }

    let mut rng = rand::thread_rng();
    for list in &mut class_indices {
        list.shuffle(&mut rng);
    }

    // k_folds 개로 고르게 분할
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

        let model = MultiClassSvm::train(
            &train_x,
            &train_y,
            num_classes,
            config.clone(),
            Some(class_names.to_vec()),
        )?;

        let preds = model.predict_batch(&test_x);
        all_y_true.extend(test_y);
        all_y_pred.extend(preds);
    }

    Ok(ClassificationMetrics::compute(
        &all_y_true,
        &all_y_pred,
        num_classes,
    ))
}
