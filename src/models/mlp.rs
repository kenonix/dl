use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::CrossEntropyLossConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    tensor::{backend::AutodiffBackend, backend::Backend, Int, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::train::EmgBatch;

/// =========================================================================
/// 🧠 75차원 sEMG 제스처 분류용 5층(은닉층 4개) 심층 MLP 아키텍처
/// =========================================================================
/// 구조:
///   [입력: 75차원] -> Linear(75, 128) -> ReLU -> Dropout(0.15)
///                 -> Linear(128, 64) -> ReLU -> Dropout(0.15)
///                 -> Linear(64, 32)  -> ReLU
///                 -> Linear(32, 16)  -> ReLU
///                 -> Linear(16, 4)   -> [출력: 4개 클래스 Logits]
/// =========================================================================

#[derive(Config, Debug)]
pub struct EmgMlpConfig {
    #[config(default = 75)]
    pub d_input: usize,
    #[config(default = 128)]
    pub d_hidden1: usize,
    #[config(default = 64)]
    pub d_hidden2: usize,
    #[config(default = 32)]
    pub d_hidden3: usize,
    #[config(default = 16)]
    pub d_hidden4: usize,
    #[config(default = 4)]
    pub num_classes: usize,
    #[config(default = 0.15)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct EmgMlpModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    linear5: Linear<B>,
    dropout: Dropout,
    activation: Relu,
}

impl EmgMlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EmgMlpModel<B> {
        EmgMlpModel {
            linear1: LinearConfig::new(self.d_input, self.d_hidden1).init(device),
            linear2: LinearConfig::new(self.d_hidden1, self.d_hidden2).init(device),
            linear3: LinearConfig::new(self.d_hidden2, self.d_hidden3).init(device),
            linear4: LinearConfig::new(self.d_hidden3, self.d_hidden4).init(device),
            linear5: LinearConfig::new(self.d_hidden4, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> EmgMlpModel<B> {
    pub fn new(device: &B::Device, input_dim: usize, num_classes: usize) -> Self {
        EmgMlpConfig {
            d_input: input_dim,
            d_hidden1: 128,
            d_hidden2: 64,
            d_hidden3: 32,
            d_hidden4: 16,
            num_classes,
            dropout: 0.15,
        }
        .init(device)
    }

    /// 2차원 텐서 입력 [Batch, 75] 순전파
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.linear3.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear4.forward(x);
        let x = self.activation.forward(x);
        self.linear5.forward(x)
    }

    /// 3차원 시계열 텐서 입력 [Batch, SeqLen, 75] 호환: 마지막 프레임 추출 후 순전파
    pub fn forward_3d(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let dims = input.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let feat_dim = dims[2];
        let last_frame = input.slice([0..batch_size, (seq_len - 1)..seq_len, 0..feat_dim]);
        let flattened = last_frame.reshape([batch_size, feat_dim]);
        self.forward(flattened)
    }

    pub fn forward_classification(
        &self,
        inputs: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(inputs);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<EmgBatch<B>, ClassificationOutput<B>> for EmgMlpModel<B> {
    fn step(&self, batch: EmgBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<EmgBatch<B>, ClassificationOutput<B>> for EmgMlpModel<B> {
    fn step(&self, batch: EmgBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
    }
}
