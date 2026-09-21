use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Lstm, LstmConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct EmgLstmModel<B: Backend> {
    lstm: Lstm<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    hidden_dim: usize,
}

impl<B: Backend> EmgLstmModel<B> {
    pub fn new(device: &B::Device, input_dim: usize, num_classes: usize) -> Self {
        let hidden_dim = 64;
        let lstm = LstmConfig::new(input_dim, hidden_dim, true).init(device);
        let linear1 = LinearConfig::new(hidden_dim, 32).init(device);
        let linear2 = LinearConfig::new(32, num_classes).init(device);

        Self {
            lstm,
            linear1,
            linear2,
            hidden_dim,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let batch_size = input.dims()[0];
        let seq_len = input.dims()[1];
        let (output, _state) = self.lstm.forward(input, None);
        let last_output = output
            .slice([0..batch_size, (seq_len - 1)..seq_len, 0..self.hidden_dim])
            .reshape([batch_size, self.hidden_dim]);
        let x = burn::tensor::activation::relu(self.linear1.forward(last_output));
        self.linear2.forward(x)
    }
}
