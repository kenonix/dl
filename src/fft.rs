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
    fn update_buffer(&mut self, raw_channels: &[f32; NUM_CHANNELS]) -> [Vec<f32>; NUM_CHANNELS] {
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

    /// 중심화신호(5) + Detrended FFT(40) + 시간도메인 6대 특징(30) = 총 75차원 통합 특징 벡터 산출
    /// 🚀 75차원 전 차원이 DC 기준선 전압의 영향을 받지 않는 완전한 DC-Invariant 벡터입니다.
    pub fn process_sample_all(&mut self, raw_channels: &[f32; NUM_CHANNELS]) -> [f32; TOTAL_FEATURES_ALL] {
        let window_data = self.update_buffer(raw_channels);
        let fft_feats = self.compute_fft(&window_data);
        let td_feats = self.compute_time_domain(&window_data);

        let mut combined = [0.0f32; TOTAL_FEATURES_ALL];
        // 1. 국소 중심화 신호 (raw - mean): 센서 착용 위치 편차 완전 상쇄
        for ch in 0..NUM_CHANNELS {
            let n = window_data[ch].len() as f32;
            let mean = window_data[ch].iter().sum::<f32>() / n;
            combined[ch] = raw_channels[ch] - mean;
        }
        // 2. Detrended FFT 진폭 스펙트럼 (40)
        combined[NUM_CHANNELS..NUM_CHANNELS + FFT_TOTAL_FEATURES].copy_from_slice(&fft_feats);
        // 3. 시간 도메인 6대 지표 (30: RMS, MAV, WL, ZC, SSC, WAMP)
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
}
