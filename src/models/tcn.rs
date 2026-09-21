use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct TcnBlock<B: Backend> {
    conv1: Conv1d<B>,
    bn1: BatchNorm<B, 1>,
    conv2: Conv1d<B>,
    bn2: BatchNorm<B, 1>,
    shortcut: Option<Conv1d<B>>,
}

impl<B: Backend> TcnBlock<B> {
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        dilation: usize,
    ) -> Self {
        let padding = dilation; // kernel_size = 3일 때 길이가 보존되는 정확한 대칭 패딩
        let conv1 = Conv1dConfig::new(in_channels, out_channels, 3)
            .with_dilation(dilation)
            .with_padding(PaddingConfig1d::Explicit(padding))
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);

        let conv2 = Conv1dConfig::new(out_channels, out_channels, 3)
            .with_dilation(dilation)
            .with_padding(PaddingConfig1d::Explicit(padding))
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);

        let shortcut = if in_channels != out_channels {
            Some(
                Conv1dConfig::new(in_channels, out_channels, 1)
                    .with_padding(PaddingConfig1d::Same)
                    .init(device),
            )
        } else {
            None
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = match &self.shortcut {
            Some(sc) => sc.forward(x.clone()),
            None => x.clone(),
        };

        let out = burn::tensor::activation::relu(self.bn1.forward(self.conv1.forward(x)));
        let out = self.bn2.forward(self.conv2.forward(out));

        burn::tensor::activation::relu(out + residual)
    }
}

/// EMG 시계열 인식용 1D Temporal Convolutional Network (TCN)
#[derive(Module, Debug)]
pub struct EmgTcnModel<B: Backend> {
    block1: TcnBlock<B>,
    block2: TcnBlock<B>,
    block3: TcnBlock<B>,
    pool: AdaptiveAvgPool1d,
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> EmgTcnModel<B> {
    pub fn new(device: &B::Device, input_dim: usize, num_classes: usize) -> Self {
        let block1 = TcnBlock::new(device, input_dim, 32, 1);
        let block2 = TcnBlock::new(device, 32, 64, 2);
        let block3 = TcnBlock::new(device, 64, 64, 4);
        let pool = AdaptiveAvgPool1dConfig::new(1).init();
        let linear1 = LinearConfig::new(64, 32).init(device);
        let linear2 = LinearConfig::new(32, num_classes).init(device);

        Self {
            block1,
            block2,
            block3,
            pool,
            linear1,
            linear2,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let batch_size = input.dims()[0];
        // [Batch, Seq_Len, Feature_Dim] -> [Batch, Feature_Dim, Seq_Len]
        let x = input.swap_dims(1, 2);

        // Dilated Residual Blocks
        let x = self.block1.forward(x);
        let x = self.block2.forward(x);
        let x = self.block3.forward(x);

        // Global Average Pooling
        let x = self.pool.forward(x).reshape([batch_size, 64]);

        // Classification Head
        let x = burn::tensor::activation::relu(self.linear1.forward(x));
        self.linear2.forward(x)
    }
}
