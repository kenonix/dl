use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityGrade {
    GradeA, // 🌟 최우수 (85 ~ 100점): 즉시 모델 학습 추천
    GradeB, // 👍 양호 (70 ~ 84점): 학습 가능
    GradeC, // ⚠️ 주의 (50 ~ 69점): 노이즈 또는 일관성 다소 불량
    GradeF, // ❌ 불량 (0 ~ 49점): 신호 분리 불가, 재수집 필수
}

impl QualityGrade {
    pub fn badge(&self) -> &'static str {
        match self {
            QualityGrade::GradeA => "🌟 GRADE A (최우수 - 즉시 학습 추천)",
            QualityGrade::GradeB => "👍 GRADE B (양호 - 안정적 학습 가능)",
            QualityGrade::GradeC => "⚠️ GRADE C (주의 - 노이즈 또는 힘 불균형)",
            QualityGrade::GradeF => "❌ GRADE F (불량 - 재수집 필수)",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DatasetQualityReport {
    pub file_name: String,
    pub total_samples: usize,
    pub num_classes: usize,
    pub class_names: Vec<String>,
    pub rest_rms: f32,
    pub action_rms_max: f32,
    pub snr_ratio: f32,
    pub snr_db: f32,
    pub saturation_rate: f32,
    pub saturated_count: usize,
    pub fisher_separability: f32,
    pub cycle_consistency: f32,
    pub total_score: f32,
    pub grade: QualityGrade,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

impl DatasetQualityReport {
    /// CSV 파일을 분석하여 종합 품질 보고서를 생성합니다.
    pub fn analyze(file_path: &str) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let mut rdr = csv::Reader::from_reader(file);

        let headers = rdr.headers().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?.clone();
        let label_idx = headers.iter().position(|h| h == "label").unwrap_or(headers.len().saturating_sub(2));
        let action_idx = headers.iter().position(|h| h == "action_name").unwrap_or(headers.len().saturating_sub(1));

        let rms_indices: Vec<usize> = headers
            .iter()
            .enumerate()
            .filter(|(_, h)| h.contains("_rms"))
            .map(|(idx, _)| idx)
            .collect();

        let raw_indices: Vec<usize> = (0..5.min(headers.len())).collect();

        let mut class_rms_map: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
        let mut class_names_map: BTreeMap<usize, String> = BTreeMap::new();
        let mut sequence_samples: Vec<(usize, f32)> = Vec::new();
        let mut saturated_count = 0usize;
        let mut total_samples = 0usize;

        for result in rdr.records() {
            let record = match result {
                Ok(r) => r,
                Err(_) => continue,
            };
            if record.len() <= label_idx {
                continue;
            }

            total_samples += 1;
            let label = record.get(label_idx).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
            let action_name = record.get(action_idx).unwrap_or("Unknown").trim().to_string();
            class_names_map.entry(label).or_insert(action_name);

            // Total AC-RMS 계산
            let total_rms: f32 = if !rms_indices.is_empty() {
                rms_indices
                    .iter()
                    .map(|&idx| record.get(idx).and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0))
                    .sum()
            } else {
                0.0
            };

            class_rms_map.entry(label).or_default().push(total_rms);
            sequence_samples.push((label, total_rms));

            // 포화(Saturation) 감지
            let mut is_sat = false;
            for &idx in &raw_indices {
                if let Some(val) = record.get(idx).and_then(|s| s.parse::<f32>().ok()) {
                    if val >= 1020.0 || val.abs() >= 450.0 {
                        is_sat = true;
                        break;
                    }
                }
            }
            if total_rms >= 800.0 {
                is_sat = true;
            }
            if is_sat {
                saturated_count += 1;
            }
        }

        if total_samples == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "유효한 데이터 행이 없습니다."));
        }

        let num_classes = class_names_map.len().max(1);
        let mut class_names = Vec::new();
        for i in 0..num_classes {
            class_names.push(class_names_map.get(&i).cloned().unwrap_or_else(|| format!("Class {}", i)));
        }

        // 1. 휴식기(label 0) RMS 평균 및 표준편차
        let rest_samples = class_rms_map.get(&0).cloned().unwrap_or_default();
        let rest_mean = if !rest_samples.is_empty() {
            rest_samples.iter().sum::<f32>() / rest_samples.len() as f32
        } else {
            1.0
        };
        let rest_std = if rest_samples.len() > 1 {
            (rest_samples.iter().map(|&x| (x - rest_mean).powi(2)).sum::<f32>() / (rest_samples.len() - 1) as f32).sqrt()
        } else {
            1.0
        };

        // 2. 비휴식 동작(label >= 1) RMS 평균 중 최댓값
        let mut action_means: Vec<f32> = Vec::new();
        let mut action_stds: Vec<f32> = Vec::new();
        for (&label, samples) in &class_rms_map {
            if label > 0 && !samples.is_empty() {
                let mean = samples.iter().sum::<f32>() / samples.len() as f32;
                let std = if samples.len() > 1 {
                    (samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (samples.len() - 1) as f32).sqrt()
                } else {
                    1.0
                };
                action_means.push(mean);
                action_stds.push(std);
            }
        }

        let action_rms_max = action_means.iter().copied().fold(0.0f32, f32::max);
        let action_std_at_max = action_means
            .iter()
            .zip(action_stds.iter())
            .max_by(|(m1, _), (m2, _)| m1.partial_cmp(m2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, &s)| s)
            .unwrap_or(1.0);

        // 3. 지표 1: SNR (Dynamic Range Ratio)
        let snr_ratio = action_rms_max / rest_mean.max(1e-4);
        let snr_db = 10.0 * (action_rms_max.powi(2) / rest_mean.powi(2).max(1e-4)).log10();

        // 4. 지표 2: 포화율 (Saturation Rate)
        let saturation_rate = (saturated_count as f32 / total_samples as f32) * 100.0;

        // 5. 지표 3: Fisher 클래스 분리도 J = |mu1 - mu0| / (std1 + std0)
        let fisher_separability = if num_classes > 1 {
            (action_rms_max - rest_mean).abs() / (action_std_at_max + rest_std).max(1e-4)
        } else {
            0.0
        };

        // 6. 지표 4: 3사이클 반복 일관성 (Cycle Consistency)
        let mut runs: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut current_label = usize::MAX;
        let mut current_block = Vec::new();

        for &(lbl, rms) in &sequence_samples {
            if lbl != current_label {
                if !current_block.is_empty() {
                    runs.push((current_label, current_block));
                }
                current_label = lbl;
                current_block = vec![rms];
            } else {
                current_block.push(rms);
            }
        }
        if !current_block.is_empty() {
            runs.push((current_label, current_block));
        }

        let mut block_means_by_class: HashMap<usize, Vec<f32>> = HashMap::new();
        for (lbl, blk) in runs {
            if blk.len() >= 20 {
                let mean = blk.iter().sum::<f32>() / blk.len() as f32;
                block_means_by_class.entry(lbl).or_default().push(mean);
            }
        }

        let mut consistency_scores: Vec<f32> = Vec::new();
        for (lbl, means) in block_means_by_class {
            if lbl > 0 && means.len() >= 2 {
                let m = means.iter().sum::<f32>() / means.len() as f32;
                let var = means.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / means.len() as f32;
                let std = var.sqrt();
                let cv = std / m.max(1e-4);
                let score = (1.0 - cv.min(1.0)) * 100.0;
                consistency_scores.push(score);
            }
        }
        let cycle_consistency = if !consistency_scores.is_empty() {
            consistency_scores.iter().sum::<f32>() / consistency_scores.len() as f32
        } else {
            80.0
        };

        // 7. 종합 점수 산출 (0 ~ 100점)
        let snr_pts = if snr_ratio >= 3.0 {
            30.0
        } else if snr_ratio >= 2.0 {
            24.0 + (snr_ratio - 2.0) * 6.0
        } else if snr_ratio >= 1.5 {
            16.0 + (snr_ratio - 1.5) * 16.0
        } else if snr_ratio >= 1.2 {
            8.0 + (snr_ratio - 1.2) * 26.6
        } else {
            0.0
        };

        let sat_pts = if saturation_rate <= 0.1 {
            25.0
        } else if saturation_rate <= 1.0 {
            20.0
        } else if saturation_rate <= 5.0 {
            10.0
        } else if saturation_rate <= 15.0 {
            3.0
        } else {
            0.0
        };

        let sep_pts = if fisher_separability >= 2.0 {
            25.0
        } else if fisher_separability >= 1.2 {
            20.0 + (fisher_separability - 1.2) * 6.25
        } else if fisher_separability >= 0.8 {
            12.0 + (fisher_separability - 0.8) * 20.0
        } else if fisher_separability >= 0.4 {
            5.0 + (fisher_separability - 0.4) * 17.5
        } else {
            0.0
        };

        let con_pts = (cycle_consistency / 100.0) * 20.0;

        let total_score = (snr_pts + sat_pts + sep_pts + con_pts).clamp(0.0, 100.0);

        let grade = if total_score >= 85.0 {
            QualityGrade::GradeA
        } else if total_score >= 70.0 {
            QualityGrade::GradeB
        } else if total_score >= 50.0 {
            QualityGrade::GradeC
        } else {
            QualityGrade::GradeF
        };

        // 문제점 및 가이드 진단
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if rest_mean > 250.0 {
            issues.push(format!("휴식 상태의 RMS 노이즈({:.1})가 비정상적으로 높습니다.", rest_mean));
            recommendations.push("전극이 피부에서 들떴거나 기준 전극(GND)이 접촉 불량일 수 있습니다. GND 전극을 단단히 밀착시키세요.".into());
        }

        if snr_ratio < 1.3 {
            issues.push(format!("동작과 휴식의 신호 차이가 거의 없습니다 (SNR Ratio: {:.2}배).", snr_ratio));
            recommendations.push("주먹을 쥘 때 더 강하게 힘을 주거나, 굴근 근육 부위에 센서 전극을 재배치하세요.".into());
        }

        if saturation_rate > 3.0 {
            issues.push(format!("신호 포화(1020 ADC) 샘플이 {:.1}% 발생하여 파형이 잘렸습니다.", saturation_rate));
            recommendations.push("센서 증폭 게인이 너무 높거나 전극이 순간적으로 피부에서 떨어집니다.".into());
        }

        if fisher_separability < 0.6 {
            issues.push(format!("동작 간의 통계적 경계가 모호합니다 (Fisher Index: {:.2}).", fisher_separability));
            recommendations.push("자세를 취하기 전 미리 힘을 준 상태에서 카운트다운을 유지하세요.".into());
        }

        if cycle_consistency < 60.0 {
            issues.push(format!("사이클별 힘의 크기 일관성이 다소 부족합니다 ({:.1}%).", cycle_consistency));
            recommendations.push("사이클마다 일정한 악력으로 주먹을 쥐도록 연습하세요.".into());
        }

        if issues.is_empty() {
            recommendations.push("모든 물리적·통계적 지표가 최상급입니다. 즉시 2번 모델 학습을 진행하세요!".into());
        }

        let file_name = Path::new(file_path).file_name().unwrap_or_default().to_string_lossy().to_string();

        Ok(Self {
            file_name,
            total_samples,
            num_classes,
            class_names,
            rest_rms: rest_mean,
            action_rms_max,
            snr_ratio,
            snr_db,
            saturation_rate,
            saturated_count,
            fisher_separability,
            cycle_consistency,
            total_score,
            grade,
            issues,
            recommendations,
        })
    }

    /// 터미널에 시각적으로 풍부한 품질 보고서 카드를 출력합니다.
    pub fn print_report(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
        println!("║               📊 [EMG 데이터셋 종합 품질 검증 성적표]                   ║");
        println!("╚══════════════════════════════════════════════════════════════════════════╝");
        println!(" 📁 파일명     : {}", self.file_name);
        println!(" 📈 총 샘플 수 : {}개 ({}프레임 @ 50Hz ≈ {:.1}초)", self.total_samples, self.total_samples, self.total_samples as f32 / 50.0);
        println!(" 🏷️ 클래스 구성: {}개 클래스 -> [{}]", self.num_classes, self.class_names.join(", "));
        println!("────────────────────────────────────────────────────────────────────────────");

        let snr_status = if self.snr_ratio >= 2.0 { "✔️ 합격" } else if self.snr_ratio >= 1.5 { "⚠️ 보통" } else { "❌ 불합격" };
        println!(
            " [1] 신호 대 잡음비 (SNR)   : {:5.2}배 ({:4.1} dB) | [휴식:{:5.1} ➔ 동작:{:5.1} RMS] [{}]",
            self.snr_ratio, self.snr_db, self.rest_rms, self.action_rms_max, snr_status
        );

        let sat_status = if self.saturation_rate <= 1.0 { "✔️ 합격" } else if self.saturation_rate <= 5.0 { "⚠️ 주의" } else { "❌ 불합격" };
        println!(
            " [2] 전극 포화율 (Clipping) : {:5.1}% ({}/{}개 포화)                    [{}]",
            self.saturation_rate, self.saturated_count, self.total_samples, sat_status
        );

        let sep_status = if self.fisher_separability >= 1.2 { "✔️ 합격" } else if self.fisher_separability >= 0.8 { "⚠️ 보통" } else { "❌ 불합격" };
        println!(
            " [3] 클래스 분리도 (Fisher) : {:5.2} (수축과 휴식의 신호 경계 명확성)      [{}]",
            self.fisher_separability, sep_status
        );

        let con_status = if self.cycle_consistency >= 75.0 { "✔️ 합격" } else if self.cycle_consistency >= 60.0 { "⚠️ 보통" } else { "❌ 주의" };
        println!(
            " [4] 3사이클 반복 일관성    : {:5.1}% (사이클 간 일정한 힘 유지율)          [{}]",
            self.cycle_consistency, con_status
        );

        println!("────────────────────────────────────────────────────────────────────────────");
        println!(" 🏆 종합 품질 점수 : {:.0}점 / 100점  [ {} ]", self.total_score, self.grade.badge());

        if !self.issues.is_empty() {
            println!("\n ⚠️ 감지된 주요 문제점:");
            for issue in &self.issues {
                println!("    • {}", issue);
            }
        }

        println!("\n 💡 분석 평가 및 조치 가이드:");
        for rec in &self.recommendations {
            println!("    ➔ {}", rec);
        }
        println!("════════════════════════════════════════════════════════════════════════════\n");
    }
}

/// 데이터 수집 직후 품질을 평가하고, 품질이 불량할 경우 원클릭 삭제 및 재수집 여부를 묻습니다.
pub fn evaluate_and_prompt_after_collection(file_path: &str) -> io::Result<bool> {
    match DatasetQualityReport::analyze(file_path) {
        Ok(report) => {
            report.print_report();

            if report.grade == QualityGrade::GradeF || report.total_score < 50.0 {
                println!("🚨 [품질 불량 경고] 본 데이터셋은 신호 노이즈로 인해 학습 시 100% 오작동합니다.");
                print!("👉 불량 데이터셋 파일('{}')을 즉시 삭제하시겠습니까? (Y/n) > ", report.file_name);
                io::stdout().flush()?;

                let mut ans = String::new();
                io::stdin().read_line(&mut ans)?;
                let ans = ans.trim();
                if ans.is_empty() || ans.eq_ignore_ascii_case("y") {
                    if let Err(e) = std::fs::remove_file(file_path) {
                        eprintln!("⚠️ 파일 삭제 실패: {}", e);
                    } else {
                        println!("🗑️ 불량 파일이 안전하게 삭제되었습니다. 전극을 조정한 후 다시 수집해주세요.");
                    }
                    return Ok(false);
                } else {
                    println!("ℹ️ 파일을 삭제하지 않고 보존합니다.");
                }
            } else if report.grade == QualityGrade::GradeC {
                println!("ℹ️ 품질이 '보통(C등급)'입니다. 학습은 가능하나, 전극 밀착 후 재수집하면 훨씬 더 높은 정확도를 얻을 수 있습니다.");
            } else {
                println!("🎉 고품질 데이터셋이 검증되었습니다! 바로 메뉴 [2번]에서 모델을 학습시키세요.");
            }
            Ok(true)
        }
        Err(e) => {
            eprintln!("⚠️ 데이터 품질 분석 중 오류 발생: {}", e);
            Ok(true)
        }
    }
}

/// 단독 품질 분석 메뉴 (저장된 모든 CSV 파일 선택 분석)
pub fn run_dataset_quality_menu() -> io::Result<()> {
    let mut found_files = Vec::new();
    for entry in glob::glob("emg_dataset_*.csv").expect("Failed glob") {
        if let Ok(path) = entry {
            found_files.push(path);
        }
    }

    found_files.sort();
    if found_files.is_empty() {
        println!("\n❌ 분석할 데이터셋 파일(emg_dataset_*.csv)이 없습니다.");
        return Ok(());
    }

    println!("\n========================================================");
    println!(" 📊 EMG 데이터셋 품질 검증 및 분석기");
    println!("========================================================");
    for (i, file) in found_files.iter().enumerate() {
        println!("  [{}] {}", i + 1, file.display());
    }
    println!("  [A] 전체 파일 일괄 품질 랭킹 비교");
    print!("선택 (기본 1번, 또는 A) > ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let trimmed = input.trim();

    if trimmed.eq_ignore_ascii_case("a") {
        println!("\n=== 🏆 전체 데이터셋 품질 랭킹 비교 ===");
        println!("{:<35} | {:<5} | {:<7} | {:<6} | {:<12}", "파일명", "점수", "SNR", "포화율", "등급");
        println!("----------------------------------------------------------------------------------");
        for file in &found_files {
            if let Ok(rep) = DatasetQualityReport::analyze(&file.to_string_lossy()) {
                println!(
                    "{:<35} | {:3.0}점  | {:5.2}배 | {:5.1}% | {}",
                    rep.file_name, rep.total_score, rep.snr_ratio, rep.saturation_rate, rep.grade.badge()
                );
            }
        }
        println!("----------------------------------------------------------------------------------");
        return Ok(());
    }

    let choice = trimmed.parse::<usize>().unwrap_or(1);
    let selected_file = if choice > 0 && choice <= found_files.len() {
        &found_files[choice - 1]
    } else {
        &found_files[0]
    };

    let report = DatasetQualityReport::analyze(&selected_file.to_string_lossy())?;
    report.print_report();

    Ok(())
}
