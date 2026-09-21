use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::collections::VecDeque;
use std::sync::Arc;

pub const NUM_CHANNELS: usize = 5;
pub const FFT_WINDOW_SIZE: usize = 16;
pub const FFT_BINS_PER_CHANNEL: usize = FFT_WINDOW_SIZE / 2; // 8 bins
pub const FFT_TOTAL_FEATURES: usize = NUM_CHANNELS * FFT_BINS_PER_CHANNEL; // 40 features
pub const TOTAL_FEATURES_WITH_RAW: usize = NUM_CHANNELS + FFT_TOTAL_FEATURES; // 45 features

// Hudgins 4대 시간 도메인 특징: RMS, MAV, WL, ZC
pub const HUDGINS_PER_CHANNEL: usize = 4;
pub const HUDGINS_TOTAL_FEATURES: usize = NUM_CHANNELS * HUDGINS_PER_CHANNEL; // 20 features
pub const TOTAL_FEATURES_ALL: usize = NUM_CHANNELS + FFT_TOTAL_FEATURES + HUDGINS_TOTAL_FEATURES; // 65 features

/// EMG 5채널 슬라이딩 윈도우 기반 실시간 특징 추출기 (FFT + Hudgins 시간 도메인 지표)
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
            let mut input: Vec<Complex<f32>> = window_data[ch]
                .iter()
                .map(|&v| Complex { re: v, im: 0.0 })
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

    /// 채널별 Hudgins 4대 특징(RMS, MAV, WL, ZC)을 계산합니다. (총 20개 값)
    pub fn compute_hudgins(&self, window_data: &[Vec<f32>; NUM_CHANNELS]) -> [f32; HUDGINS_TOTAL_FEATURES] {
        let mut hudgins_features = [0.0f32; HUDGINS_TOTAL_FEATURES];

        for ch in 0..NUM_CHANNELS {
            let samples = &window_data[ch];
            let n = samples.len() as f32;

            // 1. MAV (Mean Absolute Value) & 평균값 계산
            let mut sum_abs = 0.0f32;
            let mut sum_val = 0.0f32;
            let mut sum_sq = 0.0f32;
            for &x in samples {
                sum_abs += x.abs();
                sum_val += x;
                sum_sq += x * x;
            }
            let mav = sum_abs / n;
            let mean = sum_val / n;

            // 2. RMS (Root Mean Square)
            let rms = (sum_sq / n).sqrt();

            // 3. WL (Waveform Length)
            let mut wl = 0.0f32;
            for i in 1..samples.len() {
                wl += (samples[i] - samples[i - 1]).abs();
            }

            // 4. ZC (Zero Crossing): 신호 평균(DC)을 뺀 후 0을 교차하는 횟수
            let mut zc = 0.0f32;
            let noise_threshold = 5.0f32; // 아날로그 노이즈 방지 임계치
            for i in 1..samples.len() {
                let prev = samples[i - 1] - mean;
                let curr = samples[i] - mean;
                if ((prev > 0.0 && curr < 0.0) || (prev < 0.0 && curr > 0.0))
                    && (prev - curr).abs() >= noise_threshold
                {
                    zc += 1.0;
                }
            }

            let base = ch * HUDGINS_PER_CHANNEL;
            hudgins_features[base] = rms;
            hudgins_features[base + 1] = mav;
            hudgins_features[base + 2] = wl;
            hudgins_features[base + 3] = zc;
        }

        hudgins_features
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

    /// 원시(5) + FFT(40) + Hudgins 4대 특징(20) = 총 65차원 통합 특징 벡터 산출
    pub fn process_sample_all(&mut self, raw_channels: &[f32; NUM_CHANNELS]) -> [f32; TOTAL_FEATURES_ALL] {
        let window_data = self.update_buffer(raw_channels);
        let fft_feats = self.compute_fft(&window_data);
        let hudgins_feats = self.compute_hudgins(&window_data);

        let mut combined = [0.0f32; TOTAL_FEATURES_ALL];
        // 원시 센서값 (5)
        combined[..NUM_CHANNELS].copy_from_slice(raw_channels);
        // FFT 진폭 스펙트럼 (40)
        combined[NUM_CHANNELS..NUM_CHANNELS + FFT_TOTAL_FEATURES].copy_from_slice(&fft_feats);
        // Hudgins 시간 도메인 지표 (20)
        combined[NUM_CHANNELS + FFT_TOTAL_FEATURES..].copy_from_slice(&hudgins_feats);

        combined
    }

    /// 65차원 전체 특징 CSV 헤더 생성
    pub fn csv_header() -> String {
        let mut header = String::from("a0,a1,a2,a3,a4");
        // FFT 40
        for ch in 0..NUM_CHANNELS {
            for bin in 0..FFT_BINS_PER_CHANNEL {
                header.push_str(&format!(",a{}_f{}", ch, bin + 1));
            }
        }
        // Hudgins 20
        for ch in 0..NUM_CHANNELS {
            header.push_str(&format!(",a{}_rms,a{}_mav,a{}_wl,a{}_zc", ch, ch, ch, ch));
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
        assert_eq!(all_feats[0], 100.0);
        assert_eq!(all_feats[4], 500.0);
    }
}
