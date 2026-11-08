use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Int, Tensor},
};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::fs::File;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NormStats {
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
}

impl NormStats {
    pub fn save(&self, path: &str) {
        let file = File::create(path).expect("정규화 통계 저장 실패");
        serde_json::to_writer_pretty(file, self).expect("JSON 직렬화 실패");
    }

    pub fn load(path: &str) -> Self {
        let file = File::open(path).expect("norm_stats.json 파일을 찾을 수 없습니다.");
        serde_json::from_reader(file).expect("JSON 역직렬화 실패")
    }

    pub fn normalize(&self, features: &[f32]) -> Vec<f32> {
        features
            .iter()
            .enumerate()
            .map(|(i, &val)| (val - self.means[i]) / self.stds[i])
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct EmgItem {
    pub features: Vec<f32>,
    pub label: usize,
}

pub struct EmgDataset {
    pub items: Vec<EmgItem>,
}

impl Dataset<EmgItem> for EmgDataset {
    fn get(&self, index: usize) -> Option<EmgItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

impl EmgDataset {
    pub fn load_raw(file_path: &str) -> Vec<EmgItem> {
        let file = File::open(file_path).expect("CSV 파일을 열 수 없습니다.");
        let mut rdr = csv::Reader::from_reader(file);
        let mut raw_items = Vec::new();

        for result in rdr.records() {
            let record = result.expect("CSV 레코드 읽기 오류");
            let len = record.len();
            let features: Vec<f32> = (0..len - 2)
                .map(|i| record[i].parse::<f32>().unwrap_or(0.0))
                .collect();
            let label: usize = record[len - 2].parse().unwrap_or(0);
            raw_items.push(EmgItem { features, label });
        }
        raw_items
    }

    pub fn prepare_data(file_path: &str) -> (Self, Self, NormStats) {
        let mut raw_items = Self::load_raw(file_path);
        let num_features = raw_items[0].features.len();
        let num_samples = raw_items.len() as f32;

        let mut means = vec![0.0f32; num_features];
        let mut stds = vec![0.0f32; num_features];

        for item in &raw_items {
            for (i, val) in item.features.iter().enumerate() {
                means[i] += val;
            }
        }
        for m in &mut means {
            *m /= num_samples;
        }

        for item in &raw_items {
            for (i, val) in item.features.iter().enumerate() {
                stds[i] += (val - means[i]).powi(2);
            }
        }
        for s in &mut stds {
            *s = (*s / num_samples).sqrt().max(1e-6);
        }

        let stats = NormStats { means, stds };

        // 표준화 적용
        for item in &mut raw_items {
            item.features = stats.normalize(&item.features);
        }

        // 블록 편향 제거를 위한 셔플
        let mut rng = thread_rng();
        raw_items.shuffle(&mut rng);

        let split_idx = (raw_items.len() as f32 * 0.8) as usize;
        let test_items = raw_items.split_off(split_idx);

        (
            EmgDataset { items: raw_items },
            EmgDataset { items: test_items },
            stats,
        )
    }
}

#[derive(Clone)]
pub struct EmgBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> EmgBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct EmgBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<EmgItem, EmgBatch<B>> for EmgBatcher<B> {
    fn batch(&self, items: Vec<EmgItem>) -> EmgBatch<B> {
        let inputs = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats(item.features.as_slice(), &self.device))
            .collect();
        let inputs = Tensor::stack(inputs, 0);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_ints([item.label as i32], &self.device))
            .collect();
        let targets = Tensor::cat(targets, 0);

        EmgBatch { inputs, targets }
    }
}
