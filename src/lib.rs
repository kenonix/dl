pub mod fft;
pub mod models;
pub mod train;

pub use fft::{
    EmgFftProcessor, FFT_BINS_PER_CHANNEL, FFT_TOTAL_FEATURES, FFT_WINDOW_SIZE, NUM_CHANNELS,
    TOTAL_FEATURES_WITH_RAW,
};
pub use models::{EmgCnnLstmModel, EmgCnnModel, EmgLstmModel, ModelArchitecture};
pub use train::{
    create_sliding_windows, load_dataset, train_model, MyAutodiffBackend, MyBackend,
    TrainingConfig, TrainingProgress, NUM_CLASSES, SEQ_LEN,
};
