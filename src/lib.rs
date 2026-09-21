pub mod fft;
pub mod models;
pub mod train;

pub use fft::{
    EmgFftProcessor, FFT_BINS_PER_CHANNEL, FFT_TOTAL_FEATURES, FFT_WINDOW_SIZE,
    HUDGINS_PER_CHANNEL, HUDGINS_TOTAL_FEATURES, NUM_CHANNELS, TIME_DOMAIN_PER_CHANNEL,
    TIME_DOMAIN_TOTAL_FEATURES, TOTAL_FEATURES_ALL, TOTAL_FEATURES_WITH_RAW,
};
pub use models::{
    EmgCnnLstmModel, EmgCnnModel, EmgLstmModel, EmgTcnModel, ModelArchitecture,
};
pub use train::{
    create_sliding_windows, load_dataset, train_model, MyAutodiffBackend, MyBackend,
    NormalizationStats, TrainingConfig, TrainingProgress, NUM_CLASSES, SEQ_LEN,
};
