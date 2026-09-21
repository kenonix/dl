use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct EmgCnnModel<B: Backend> {
    conv1: Conv1d<B>,
    bn1: BatchNorm<B, 1>,
    conv2: Conv1d<B>,
    bn2: BatchNorm<B, 1>,
    pool: AdaptiveAvgPool1d,
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> EmgCnnModel<B> {
    pub fn new(device: &B::Device, input_dim: usize, num_classes: usize) -> Self {
        let conv1 = Conv1dConfig::new(input_dim, 32, 5)
            .with_padding(PaddingConfig1d::Same)
            .init(device);
        let bn1 = BatchNormConfig::new(32).init(device);
        let conv2 = Conv1dConfig::new(32, 64, 5)
            .with_padding(PaddingConfig1d::Same)
            .init(device);
        let bn2 = BatchNormConfig::new(64).init(device);
        let pool = AdaptiveAvgPool1dConfig::new(1).init();
        let linear1 = LinearConfig::new(64, 32).init(device);
        let linear2 = LinearConfig::new(32, num_classes).init(device);

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            pool,
            linear1,
            linear2,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let batch_size = input.dims()[0];
        // [Batch, Seq_Len, Feature_Dim] -> [Batch, Feature_Dim, Seq_Len]
        let x = input.swap_dims(1, 2);
        let x = burn::tensor::activation::relu(self.bn1.forward(self.conv1.forward(x)));
        let x = burn::tensor::activation::relu(self.bn2.forward(self.conv2.forward(x)));
        // Pool: [Batch, 64, 1] -> Reshape: [Batch, 64]
        let x = self.pool.forward(x).reshape([batch_size, 64]);
        let x = burn::tensor::activation::relu(self.linear1.forward(x));
        self.linear2.forward(x)
    }
}
