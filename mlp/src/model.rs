use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::CrossEntropyLossConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    tensor::{backend::AutodiffBackend, backend::Backend, Int, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::dataset::EmgBatch;

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 75)]
    pub d_input: usize,
    #[config(default = 64)]
    pub d_hidden1: usize,
    #[config(default = 32)]
    pub d_hidden2: usize,
    #[config(default = 4)]
    pub num_classes: usize,
    #[config(default = 0.2)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    dropout: Dropout,
    activation: Relu,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(self.d_input, self.d_hidden1).init(device),
            linear2: LinearConfig::new(self.d_hidden1, self.d_hidden2).init(device),
            linear3: LinearConfig::new(self.d_hidden2, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        self.linear3.forward(x)
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

impl<B: AutodiffBackend> TrainStep<EmgBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: EmgBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<EmgBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: EmgBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
    }
}
