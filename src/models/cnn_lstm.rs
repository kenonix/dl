use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig, Lstm, LstmConfig, PaddingConfig1d};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct EmgCnnLstmModel<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
    lstm: Lstm<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    hidden_dim: usize,
}

impl<B: Backend> EmgCnnLstmModel<B> {
    pub fn new(device: &B::Device, input_dim: usize, num_classes: usize) -> Self {
        let conv_channels = 32;
        let hidden_dim = 64;

        let conv1 = Conv1dConfig::new(input_dim, conv_channels, 5)
            .with_padding(PaddingConfig1d::Same)
            .init(device);
        let conv2 = Conv1dConfig::new(conv_channels, conv_channels, 3)
            .with_padding(PaddingConfig1d::Same)
            .init(device);
        let lstm = LstmConfig::new(conv_channels, hidden_dim, true).init(device);
        let linear1 = LinearConfig::new(hidden_dim, 32).init(device);
        let linear2 = LinearConfig::new(32, num_classes).init(device);

        Self {
            conv1,
            conv2,
            lstm,
            linear1,
            linear2,
            hidden_dim,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let batch_size = input.dims()[0];
        let seq_len = input.dims()[1];

        // 1. Conv1d: [Batch, Seq_Len, Feature_Dim] -> [Batch, Feature_Dim, Seq_Len]
        let x = input.swap_dims(1, 2);
        let x = burn::tensor::activation::relu(self.conv1.forward(x));
        let x = burn::tensor::activation::relu(self.conv2.forward(x));

        // 2. LSTM 입력 형태로 변환: [Batch, Conv_Channels, Seq_Len] -> [Batch, Seq_Len, Conv_Channels]
        let x = x.swap_dims(1, 2);

        // 3. LSTM 통과
        let (output, _state) = self.lstm.forward(x, None);
        let last_output = output
            .slice([0..batch_size, (seq_len - 1)..seq_len, 0..self.hidden_dim])
            .reshape([batch_size, self.hidden_dim]);

        // 4. 분류 헤드
        let x = burn::tensor::activation::relu(self.linear1.forward(last_output));
        self.linear2.forward(x)
    }
}
