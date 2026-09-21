use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::collections::VecDeque;
use std::sync::Arc;

pub const NUM_CHANNELS: usize = 5;
pub const FFT_WINDOW_SIZE: usize = 16;
pub const FFT_BINS_PER_CHANNEL: usize = FFT_WINDOW_SIZE / 2; // 8 bins
pub const FFT_TOTAL_FEATURES: usize = NUM_CHANNELS * FFT_BINS_PER_CHANNEL; // 40 features
pub const TOTAL_FEATURES_WITH_RAW: usize = NUM_CHANNELS + FFT_TOTAL_FEATURES; // 45 features

/// EMG 5채널 슬라이딩 윈도우 기반 실시간 FFT 프로세서
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

    /// 새로운 5채널 센서 샘플을 입력받아, 채널별 FFT 크기(Magnitude) 스펙트럼(40개 값)을 계산합니다.
    pub fn process_sample(&mut self, raw_channels: &[f32; NUM_CHANNELS]) -> [f32; FFT_TOTAL_FEATURES] {
        let mut fft_features = [0.0f32; FFT_TOTAL_FEATURES];

        for ch in 0..NUM_CHANNELS {
            if self.buffers[ch].len() == FFT_WINDOW_SIZE {
                self.buffers[ch].pop_front();
            }
            self.buffers[ch].push_back(raw_channels[ch]);

            // 버퍼가 아직 가득 차지 않은 경우, 현재 채널의 마지막 값으로 패딩
            let mut input: Vec<Complex<f32>> = Vec::with_capacity(FFT_WINDOW_SIZE);
            let fill_val = self.buffers[ch].front().copied().unwrap_or(raw_channels[ch]);
            let missing = FFT_WINDOW_SIZE.saturating_sub(self.buffers[ch].len());
            for _ in 0..missing {
                input.push(Complex { re: fill_val, im: 0.0 });
            }
            for &val in &self.buffers[ch] {
                input.push(Complex { re: val, im: 0.0 });
            }

            // FFT 수행
            self.fft.process(&mut input);

            // 주파수 bin 1부터 (FFT_WINDOW_SIZE / 2)까지의 magnitude 계산 (DC 성분 제외한 교류 주파수 성분)
            let base_idx = ch * FFT_BINS_PER_CHANNEL;
            for bin in 0..FFT_BINS_PER_CHANNEL {
                let c = input[bin + 1]; // bin 1 ~ 8
                let mag = (c.re * c.re + c.im * c.im).sqrt();
                fft_features[base_idx + bin] = mag;
            }
        }

        fft_features
    }

    /// 원시 센서값(5개)과 FFT 특징값(40개)을 합쳐 45차원의 통합 특징 벡터를 반환합니다.
    pub fn process_sample_combined(&mut self, raw_channels: &[f32; NUM_CHANNELS]) -> [f32; TOTAL_FEATURES_WITH_RAW] {
        let fft_feats = self.process_sample(raw_channels);
        let mut combined = [0.0f32; TOTAL_FEATURES_WITH_RAW];
        
        // 앞쪽 5개는 원시 센서값
        combined[..NUM_CHANNELS].copy_from_slice(raw_channels);
        // 뒤쪽 40개는 FFT 주파수 크기 스펙트럼
        combined[NUM_CHANNELS..].copy_from_slice(&fft_feats);

        combined
    }

    /// CSV 헤더 생성을 위한 유틸리티 함수
    pub fn csv_header() -> String {
        let mut header = String::from("a0,a1,a2,a3,a4");
        for ch in 0..NUM_CHANNELS {
            for bin in 0..FFT_BINS_PER_CHANNEL {
                header.push_str(&format!(",a{}_f{}", ch, bin + 1));
            }
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
        assert_eq!(combined[0], 100.0);
        assert_eq!(combined[4], 500.0);
    }
}
