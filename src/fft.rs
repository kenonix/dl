use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::collections::VecDeque;
use std::sync::Arc;

pub const NUM_CHANNELS: usize = 5;
pub const FFT_WINDOW_SIZE: usize = 16;
pub const FFT_BINS_PER_CHANNEL: usize = FFT_WINDOW_SIZE / 2; // 8 bins
pub const FFT_TOTAL_FEATURES: usize = NUM_CHANNELS * FFT_BINS_PER_CHANNEL; // 40 features
pub const TOTAL_FEATURES_WITH_RAW: usize = NUM_CHANNELS + FFT_TOTAL_FEATURES; // 45 features

// 시간 도메인 6대 특징: AC-RMS, AC-MAV, WL, ZC, SSC, WAMP
pub const TIME_DOMAIN_PER_CHANNEL: usize = 6;
pub const TIME_DOMAIN_TOTAL_FEATURES: usize = NUM_CHANNELS * TIME_DOMAIN_PER_CHANNEL; // 30 features
pub const TOTAL_FEATURES_ALL: usize = NUM_CHANNELS + FFT_TOTAL_FEATURES + TIME_DOMAIN_TOTAL_FEATURES; // 75 features

// 🚀 35차원 경량 시간 도메인 전용 특징: 원시 평활 5 + 시간 도메인 6대 지표 30
pub const TD_CORE_FEATURES: usize = NUM_CHANNELS + TIME_DOMAIN_TOTAL_FEATURES; // 35 features

// 호환성 별칭
pub const HUDGINS_PER_CHANNEL: usize = TIME_DOMAIN_PER_CHANNEL;
pub const HUDGINS_TOTAL_FEATURES: usize = TIME_DOMAIN_TOTAL_FEATURES;

/// EMG 5채널 슬라이딩 윈도우 기반 실시간 특징 추출기 (FFT + 시간 도메인 6대 지표)
pub struct EmgFftProcessor {
    buffers: [VecDeque<f32>; NUM_CHANNELS],
    fft: Arc<dyn Fft<f32>>,
}

impl EmgFftProcessor {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_WINDOW_SIZE);
        let buffers = [
            VecDeque::with_capacity(FFT_WINDOW_SIZE),
            VecDeque::with_capacity(FFT_WINDOW_SIZE),
            VecDeque::with_capacity(FFT_WINDOW_SIZE),
            VecDeque::with_capacity(FFT_WINDOW_SIZE),
            VecDeque::with_capacity(FFT_WINDOW_SIZE),
        ];

        Self { buffers, fft }
    }

    /// 채널별 슬라이딩 윈도우를 업데이트하고 현재 윈도우 버퍼(16개)를 반환합니다.
    pub fn update_buffer(&mut self, raw_channels: &[f32; NUM_CHANNELS]) -> [Vec<f32>; NUM_CHANNELS] {
        let mut window_data: [Vec<f32>; NUM_CHANNELS] = Default::default();

        for ch in 0..NUM_CHANNELS {
            if self.buffers[ch].len() == FFT_WINDOW_SIZE {
                self.buffers[ch].pop_front();
            }
            self.buffers[ch].push_back(raw_channels[ch]);

            let fill_val = self.buffers[ch].front().copied().unwrap_or(raw_channels[ch]);
            let missing = FFT_WINDOW_SIZE.saturating_sub(self.buffers[ch].len());
            let mut ch_vec = Vec::with_capacity(FFT_WINDOW_SIZE);
            for _ in 0..missing {
                ch_vec.push(fill_val);
            }
            for &val in &self.buffers[ch] {
                ch_vec.push(val);
            }
            window_data[ch] = ch_vec;
        }

        window_data
    }

    /// 새로운 5채널 센서 샘플을 입력받아, 채널별 FFT 크기(Magnitude) 스펙트럼(40개 값)을 계산합니다.
    pub fn process_sample(&mut self, raw_channels: &[f32; NUM_CHANNELS]) -> [f32; FFT_TOTAL_FEATURES] {
        let window_data = self.update_buffer(raw_channels);
        self.compute_fft(&window_data)
    }

    fn compute_fft(&self, window_data: &[Vec<f32>; NUM_CHANNELS]) -> [f32; FFT_TOTAL_FEATURES] {
        let mut fft_features = [0.0f32; FFT_TOTAL_FEATURES];

        for ch in 0..NUM_CHANNELS {
            let samples = &window_data[ch];
            let n = samples.len() as f32;
            let mean = samples.iter().sum::<f32>() / n;

            // 🚀 Detrended FFT: 윈도우 평균(DC)을 빼서 저주파/고주파 전 대역으로의 직류 누설 차단
            let mut input: Vec<Complex<f32>> = samples
                .iter()
                .map(|&v| Complex { re: v - mean, im: 0.0 })
                .collect();

            self.fft.process(&mut input);

            let base_idx = ch * FFT_BINS_PER_CHANNEL;
            for bin in 0..FFT_BINS_PER_CHANNEL {
                let c = input[bin + 1]; // bin 1 ~ 8
                let mag = (c.re * c.re + c.im * c.im).sqrt();
                fft_features[base_idx + bin] = mag;
            }
        }

        fft_features
    }

    /// 채널별 시간 도메인 6대 특징(AC-RMS, AC-MAV, WL, ZC, SSC, WAMP)을 계산합니다. (총 30개 값)
    /// 🚀 모든 진폭 지표를 직류(DC) 편향과 완전히 독립적인 교류(AC) 및 차분 성분으로 산출합니다.
    pub fn compute_time_domain(&self, window_data: &[Vec<f32>; NUM_CHANNELS]) -> [f32; TIME_DOMAIN_TOTAL_FEATURES] {
        let mut td_features = [0.0f32; TIME_DOMAIN_TOTAL_FEATURES];

        for ch in 0..NUM_CHANNELS {
            let samples = &window_data[ch];
            let n = samples.len() as f32;

            // 1. 국소 윈도우 평균값(DC) 계산
            let mut sum_val = 0.0f32;
            for &x in samples {
                sum_val += x;
            }
            let mean = sum_val / n;

            // 2. AC 성분 (x - mean) 기반 MAV 및 RMS 계산
            let mut sum_ac_abs = 0.0f32;
            let mut sum_ac_sq = 0.0f32;
            for &x in samples {
                let ac = x - mean;
                sum_ac_abs += ac.abs();
                sum_ac_sq += ac * ac;
            }
            let ac_mav = sum_ac_abs / n;
            let ac_rms = (sum_ac_sq / n).sqrt();

            // 3. WL (Waveform Length): 차분 신호이므로 이미 DC 불변
            let mut wl = 0.0f32;
            for i in 1..samples.len() {
                wl += (samples[i] - samples[i - 1]).abs();
            }

            // 4. ZC (Zero Crossing): 신호 평균(DC)을 뺀 후 0을 교차하는 횟수
            let mut zc = 0.0f32;
            let zc_threshold = 3.0f32; // 아날로그 노이즈 방지 임계치
            for i in 1..samples.len() {
                let prev = samples[i - 1] - mean;
                let curr = samples[i] - mean;
                if ((prev > 0.0 && curr < 0.0) || (prev < 0.0 && curr > 0.0))
                    && (prev - curr).abs() >= zc_threshold
                {
                    zc += 1.0;
                }
            }

            // 5. SSC (Slope Sign Change): 기울기 부호 역전 횟수 (주파수/발화율 지표)
            let mut ssc = 0.0f32;
            let ssc_threshold_sq = 9.0f32; // (>= 3.0 ADC counts)
            for i in 1..(samples.len() - 1) {
                let diff_prev = samples[i] - samples[i - 1];
                let diff_next = samples[i] - samples[i + 1];
                if diff_prev * diff_next >= ssc_threshold_sq {
                    ssc += 1.0;
                }
            }

            // 6. WAMP (Willison Amplitude): 수축 임계치(10.0 ADC)를 초과하는 전이 횟수 (근수축 밀도 지표)
            let mut wamp = 0.0f32;
            let wamp_threshold = 10.0f32;
            for i in 1..samples.len() {
                if (samples[i] - samples[i - 1]).abs() >= wamp_threshold {
                    wamp += 1.0;
                }
            }

            let base = ch * TIME_DOMAIN_PER_CHANNEL;
            td_features[base] = ac_rms;
            td_features[base + 1] = ac_mav;
            td_features[base + 2] = wl;
            td_features[base + 3] = zc;
            td_features[base + 4] = ssc;
            td_features[base + 5] = wamp;
        }

        td_features
    }

    pub fn compute_hudgins(&self, window_data: &[Vec<f32>; NUM_CHANNELS]) -> [f32; TIME_DOMAIN_TOTAL_FEATURES] {
        self.compute_time_domain(window_data)
    }

    /// 원시(5) + FFT(40) 결합 (45차원)
    pub fn process_sample_combined(&mut self, raw_channels: &[f32; NUM_CHANNELS]) -> [f32; TOTAL_FEATURES_WITH_RAW] {
        let window_data = self.update_buffer(raw_channels);
        let fft_feats = self.compute_fft(&window_data);

        let mut combined = [0.0f32; TOTAL_FEATURES_WITH_RAW];
        combined[..NUM_CHANNELS].copy_from_slice(raw_channels);
        combined[NUM_CHANNELS..].copy_from_slice(&fft_feats);
        combined
    }

    /* =========================================================================================================
     * ⭐⭐⭐ [핵심 전처리] 5채널 원시 신호 (Raw sEMG) -> 75차원 DC-Invariant 통합 특징 벡터 변환 ⭐⭐⭐
     * =========================================================================================================
     * 5채널 sEMG 센서 원시 입력 x_raw = [ch0, ch1, ch2, ch3, ch4]을 슬라이딩 윈도우(N=16) 기반으로
     * 피부 임피던스 및 착용 위치 편차(DC 오프셋)를 완전히 제거하고 주파수/시간 복합 75차원으로 변환합니다.
     *
     * ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
     * │ [1] 채널별 국소 중심화 신호 (Local Centering / Baseline Invariant) : 5차원 (채널당 1개)                │
     * │     수식: x̃_ch = x_raw[ch] - μ_ch                                                                     │
     * │     (여기서 μ_ch = (1/N) * ∑_{i=0}^{N-1} x_i, 국소 윈도우 N=16의 직류 평균값)                                │
     * │     -> 팔 착용 위치나 땀/피부 저항에 의해 기준선 전압이 달라져도 항상 0을 기준으로 교류(AC) 성분만 남김.     │
     * ├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     * │ [2] Detrended FFT 진폭 스펙트럼 (Magnitude Frequency Bins) : 40차원 (5채널 × 8 Bins)                   │
     * │     직류(DC) 성분이 제거된 윈도우 신호 x'_n = x_n - μ_ch (n=0..15) 에 대해 16-point Forward FFT 수행:   │
     * │     수식: X_ch[k] = ∑_{n=0}^{N-1} x'_n * exp(-j * 2π * n * k / N)                                     │
     * │     진폭: M_k = |X_ch[k]| = √( Re(X_ch[k])² + Im(X_ch[k])² )                                          │
     * │     (k = 1, 2, ..., 8 : DC bin인 k=0을 제외한 유의미한 고주파/근육 수축 대역 8개 bin 추출)                    │
     * ├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
     * │ [3] 시간 도메인 6대 핵심 특징 (Time-Domain Kinematic Features) : 30차원 (5채널 × 6지표)                 │
     * │     ① AC-RMS (Root Mean Square, 순수 교류 실효치 / 근육 수축 총에너지):                                 │
     * │        수식: RMS_ch = √( (1/N) * ∑_{n=0}^{N-1} (x_n - μ_ch)² )                                         │
     * │     ② AC-MAV (Mean Absolute Value, 평균 절대 진폭):                                                   │
     * │        수식: MAV_ch = (1/N) * ∑_{n=0}^{N-1} |x_n - μ_ch|                                               │
     * │     ③ WL (Waveform Length, 파형 복잡도 및 누적 변동 길이):                                             │
     * │        수식: WL_ch = ∑_{n=1}^{N-1} |x_n - x_{n-1}|                                                     │
     * │     ④ ZC (Zero Crossing, 직류 기준선 교차 빈도 - 운동 단위 발화율):                                     │
     * │        수식: ZC_ch = ∑_{n=1}^{N-1} I( (x_n - μ_ch)(x_{n-1} - μ_ch) < 0 ∧ |x_n - x_{n-1}| ≥ ε_zc )       │
     * │              (노이즈 방지 임계치 ε_zc = 3.0 ADC counts)                                                │
     * │     ⑤ SSC (Slope Sign Change, 기울기 부호 역전수 - 신호 고조파 주파수 특성):                           │
     * │        수식: SSC_ch = ∑_{n=1}^{N-2} I( (x_n - x_{n-1})(x_n - x_{n+1}) ≥ ε_ssc² )                       │
     * │              (노이즈 방지 임계치 ε_ssc = 3.0 ADC counts, ε_ssc² = 9.0)                                 │
     * │     ⑥ WAMP (Willison Amplitude, 윌리슨 진폭 역치수 - 활성 운동 단위 밀도):                              │
     * │        수식: WAMP_ch = ∑_{n=1}^{N-1} I( |x_n - x_{n-1}| ≥ θ_wamp )                                     │
     * │              (수축 임계치 θ_wamp = 10.0 ADC counts)                                                    │
     * └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
     *  총 특징 차원 = 5 (중심화 신호) + 40 (FFT 주파수) + 30 (시간 도메인) = 75차원
     * ========================================================================================================= */
    pub fn process_sample_all(&mut self, raw_channels: &[f32; NUM_CHANNELS]) -> [f32; TOTAL_FEATURES_ALL] {
        let window_data = self.update_buffer(raw_channels);
        let fft_feats = self.compute_fft(&window_data);
        let td_feats = self.compute_time_domain(&window_data);

        let mut combined = [0.0f32; TOTAL_FEATURES_ALL];
        // 1. 국소 중심화 평활 신호 (x̃_smooth = 3-tap LPF on (raw - mean)):
        //    센서 착용 편차를 상쇄하면서도 고주파 교류 극성 반전(Sign Alternation)에 의한 예측값 튐을 완화 (5차원)
        for ch in 0..NUM_CHANNELS {
            let n = window_data[ch].len() as f32;
            let mean = window_data[ch].iter().sum::<f32>() / n;
            let buf_len = window_data[ch].len();
            let smoothed = if buf_len >= 3 {
                let s0 = window_data[ch][buf_len - 1] - mean;
                let s1 = window_data[ch][buf_len - 2] - mean;
                let s2 = window_data[ch][buf_len - 3] - mean;
                0.5 * s0 + 0.3 * s1 + 0.2 * s2
            } else {
                raw_channels[ch] - mean
            };
            combined[ch] = smoothed;
        }
        // 2. Detrended FFT 진폭 스펙트럼 (40차원: 5ch × 8 bins)
        combined[NUM_CHANNELS..NUM_CHANNELS + FFT_TOTAL_FEATURES].copy_from_slice(&fft_feats);
        // 3. 시간 도메인 6대 지표 (30차원: RMS, MAV, WL, ZC, SSC, WAMP)
        combined[NUM_CHANNELS + FFT_TOTAL_FEATURES..].copy_from_slice(&td_feats);

        combined
    }

    /// 75차원 특징 벡터에서 5개 채널의 총 AC-RMS 에너지를 합산합니다.
    pub fn extract_total_ac_rms(feats: &[f32; TOTAL_FEATURES_ALL]) -> f32 {
        let mut sum = 0.0f32;
        for ch in 0..NUM_CHANNELS {
            let idx = NUM_CHANNELS + FFT_TOTAL_FEATURES + ch * TIME_DOMAIN_PER_CHANNEL;
            sum += feats[idx];
        }
        sum
    }

    /// 🚀 35차원 경량 특징 추출: FFT(40차원)를 완전히 생략하고
    /// 5채널 국소 중심화 평활 신호(5) + 시간 도메인 6대 지표(30)만 추출합니다.
    /// 연산량이 대폭 절감되고 다중공선성이 제거되어 온디바이스 및 고신뢰성 다중 파지에 최적화됩니다.
    pub fn process_sample_td35(&mut self, raw_channels: &[f32; NUM_CHANNELS]) -> [f32; TD_CORE_FEATURES] {
        let window_data = self.update_buffer(raw_channels);
        let td_feats = self.compute_time_domain(&window_data);

        let mut combined = [0.0f32; TD_CORE_FEATURES];
        // 1. 국소 중심화 평활 신호 (5차원)
        for ch in 0..NUM_CHANNELS {
            let n = window_data[ch].len() as f32;
            let mean = window_data[ch].iter().sum::<f32>() / n;
            let buf_len = window_data[ch].len();
            let smoothed = if buf_len >= 3 {
                let s0 = window_data[ch][buf_len - 1] - mean;
                let s1 = window_data[ch][buf_len - 2] - mean;
                let s2 = window_data[ch][buf_len - 3] - mean;
                0.5 * s0 + 0.3 * s1 + 0.2 * s2
            } else {
                raw_channels[ch] - mean
            };
            combined[ch] = smoothed;
        }
        // 2. 시간 도메인 6대 지표 (30차원)
        combined[NUM_CHANNELS..].copy_from_slice(&td_feats);

        combined
    }

    /// 35차원 특징 벡터에서 5개 채널의 총 AC-RMS 에너지를 합산합니다.
    pub fn extract_total_ac_rms_td35(feats: &[f32; TD_CORE_FEATURES]) -> f32 {
        let mut sum = 0.0f32;
        for ch in 0..NUM_CHANNELS {
            let idx = NUM_CHANNELS + ch * TIME_DOMAIN_PER_CHANNEL;
            sum += feats[idx];
        }
        sum
    }

    /// 35차원 또는 75차원 슬라이스에서 5개 채널 각각의 AC-RMS를 추출합니다.
    pub fn extract_channel_rms(feats: &[f32]) -> [f32; NUM_CHANNELS] {
        let mut rms_arr = [0.0f32; NUM_CHANNELS];
        if feats.len() >= TOTAL_FEATURES_ALL {
            for ch in 0..NUM_CHANNELS {
                let idx = NUM_CHANNELS + FFT_TOTAL_FEATURES + ch * TIME_DOMAIN_PER_CHANNEL;
                rms_arr[ch] = feats[idx];
            }
        } else if feats.len() >= TD_CORE_FEATURES {
            for ch in 0..NUM_CHANNELS {
                let idx = NUM_CHANNELS + ch * TIME_DOMAIN_PER_CHANNEL;
                rms_arr[ch] = feats[idx];
            }
        } else if feats.len() >= NUM_CHANNELS {
            for ch in 0..NUM_CHANNELS {
                rms_arr[ch] = feats[ch].abs();
            }
        }
        rms_arr
    }

    /// 채널별 RMS 에너지를 기반으로 악력 및 근피로에 불변인 공간 상대 에너지 비율 [p_0, ..., p_4]을 산출합니다.
    /// 수식: p_ch = RMS_ch / (sum(RMS) + 1e-4)
    pub fn compute_relative_spatial_distribution(channel_rms: &[f32; NUM_CHANNELS]) -> [f32; NUM_CHANNELS] {
        let sum: f32 = channel_rms.iter().sum();
        let denom = sum.max(1e-4);
        let mut dist = [0.0f32; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            dist[ch] = channel_rms[ch] / denom;
        }
        dist
    }

    /// 35차원 경량 특징 CSV 헤더 생성
    pub fn csv_header_td35() -> String {
        let mut header = String::from("a0,a1,a2,a3,a4");
        for ch in 0..NUM_CHANNELS {
            header.push_str(&format!(",a{}_rms,a{}_mav,a{}_wl,a{}_zc,a{}_ssc,a{}_wamp", ch, ch, ch, ch, ch, ch));
        }
        header.push_str(",label,action_name");
        header
    }

    /// 75차원 전체 특징 CSV 헤더 생성
    pub fn csv_header() -> String {
        let mut header = String::from("a0,a1,a2,a3,a4");
        // FFT 40
        for ch in 0..NUM_CHANNELS {
            for bin in 0..FFT_BINS_PER_CHANNEL {
                header.push_str(&format!(",a{}_f{}", ch, bin + 1));
            }
        }
        // Time-Domain 30
        for ch in 0..NUM_CHANNELS {
            header.push_str(&format!(",a{}_rms,a{}_mav,a{}_wl,a{}_zc,a{}_ssc,a{}_wamp", ch, ch, ch, ch, ch, ch));
        }
        header.push_str(",label,action_name");
        header
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_processor() {
        let mut proc = EmgFftProcessor::new();
        let sample = [100.0, 200.0, 300.0, 400.0, 500.0];
        let feats = proc.process_sample(&sample);
        assert_eq!(feats.len(), FFT_TOTAL_FEATURES);

        let combined = proc.process_sample_combined(&sample);
        assert_eq!(combined.len(), TOTAL_FEATURES_WITH_RAW);

        let all_feats = proc.process_sample_all(&sample);
        assert_eq!(all_feats.len(), TOTAL_FEATURES_ALL);
        assert_eq!(all_feats[0], 0.0);
    }

    #[test]
    fn test_dc_invariance() {
        // 동일한 AC 진동 신호에 각각 다른 DC 오프셋(300 vs 700)을 더해도 75차원 특징값이 동일해야 함
        let mut proc1 = EmgFftProcessor::new();
        let mut proc2 = EmgFftProcessor::new();

        let mut out1 = [0.0; TOTAL_FEATURES_ALL];
        let mut out2 = [0.0; TOTAL_FEATURES_ALL];

        for i in 0..25 {
            let ac = (i as f32 * 0.5).sin() * 50.0;
            let sample_dc300 = [300.0 + ac, 300.0 + ac, 300.0 + ac, 300.0 + ac, 300.0 + ac];
            let sample_dc700 = [700.0 + ac, 700.0 + ac, 700.0 + ac, 700.0 + ac, 700.0 + ac];

            out1 = proc1.process_sample_all(&sample_dc300);
            out2 = proc2.process_sample_all(&sample_dc700);
        }

        // 75차원 전체가 DC 오프셋 차이에도 불구하고 오차 0.05 이내로 일치해야 함
        for k in 0..TOTAL_FEATURES_ALL {
            let diff = (out1[k] - out2[k]).abs();
            assert!(
                diff < 0.05,
                "Feature index {} differed by {}: out1={}, out2={}",
                k,
                diff,
                out1[k],
                out2[k]
            );
        }
    }

    #[test]
    fn test_td35_features() {
        let mut proc = EmgFftProcessor::new();
        let sample = [150.0, 250.0, 350.0, 450.0, 550.0];
        let feats35 = proc.process_sample_td35(&sample);
        assert_eq!(feats35.len(), TD_CORE_FEATURES); // 35 dimensions

        let total_rms = EmgFftProcessor::extract_total_ac_rms_td35(&feats35);
        assert!(total_rms >= 0.0);

        let ch_rms = EmgFftProcessor::extract_channel_rms(&feats35);
        assert_eq!(ch_rms.len(), NUM_CHANNELS);

        // 스케일 불변성 검증: 2배 강한 악력이어도 상대 공간 비율은 동일해야 함
        let ch_rms_doubled = [ch_rms[0] * 2.0, ch_rms[1] * 2.0, ch_rms[2] * 2.0, ch_rms[3] * 2.0, ch_rms[4] * 2.0];
        let dist1 = EmgFftProcessor::compute_relative_spatial_distribution(&ch_rms);
        let dist2 = EmgFftProcessor::compute_relative_spatial_distribution(&ch_rms_doubled);
        for ch in 0..NUM_CHANNELS {
            assert!((dist1[ch] - dist2[ch]).abs() < 1e-4);
        }
    }
}
