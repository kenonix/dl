pub mod fft;
pub mod lda;
pub mod models;
pub mod quality;
pub mod svm;
pub mod train;
pub mod visualizer;

pub use quality::{
    evaluate_and_prompt_after_collection, run_dataset_quality_menu, DatasetQualityReport,
    QualityGrade,
};

pub use fft::{
    EmgFftProcessor, FFT_BINS_PER_CHANNEL, FFT_TOTAL_FEATURES, FFT_WINDOW_SIZE,
    HUDGINS_PER_CHANNEL, HUDGINS_TOTAL_FEATURES, NUM_CHANNELS, TD_CORE_FEATURES,
    TIME_DOMAIN_PER_CHANNEL, TIME_DOMAIN_TOTAL_FEATURES, TOTAL_FEATURES_ALL,
    TOTAL_FEATURES_WITH_RAW,
};
pub use lda::{save_lda4d_csv, slice_stable_3s_dataset, LdaModel};
pub use models::{
    EmgCnnLstmModel, EmgCnnModel, EmgLstmModel, EmgMlpConfig, EmgMlpModel, EmgModularMlpConfig,
    EmgModularMlpModel, EmgTcnModel, ModelArchitecture, RealTimeRingBuffer, SpatioTemporalMlpBlock,
    SwiGLUChannelBlock, TemporalMixingBlock, WarmupCosineLrScheduler, MODULAR_MLP_WINDOW_SIZE,
};
pub use svm::{
    cross_validate_svm, BinaryClassifier, BinaryLinearSvm, BinaryRbfSvm, ClassificationMetrics,
    MultiClassSvm, SvmConfig, SvmKernel,
};
pub use train::{
    create_sliding_windows, load_dataset, load_mlp_dataset, load_modular_mlp_dataset,
    load_seq_dataset, train_cnn_model, train_cnnlstm_model, train_lstm_model, train_mlp_model,
    train_model, train_modular_mlp_model, train_tcn_model, EmgBatch, EmgBatcher, EmgDataset,
    EmgItem, EmgSeqBatch, EmgSeqBatcher, EmgSeqDataset, EmgSeqItem, MyAutodiffBackend, MyBackend,
    NormalizationStats, TrainingConfig, TrainingProgress, NUM_CLASSES, SEQ_LEN,
};
pub use visualizer::{
    compute_pca_3d, generate_pca_3d_plot, generate_physical_3d_plot,
    generate_trajectory_3d_plot, load_csv_records, run_3d_visualization_menu,
    EmgCsvRecord, VisualizationMode,
};

pub const SERIAL_CONFIG_FILE: &str = "serial_config.json";
pub const DEFAULT_SERIAL_PORT: &str = "/dev/ttyACM0";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerialConfig {
    pub port_name: String,
}

impl Default for SerialConfig {
    fn default() -> Self {
        Self {
            port_name: DEFAULT_SERIAL_PORT.to_string(),
        }
    }
}

/// 저장된 시리얼 포트 경로를 불러옵니다. 파일이 없으면 기본값("/dev/ttyACM0")을 반환합니다.
pub fn load_serial_port_name() -> String {
    if let Ok(file) = std::fs::File::open(SERIAL_CONFIG_FILE) {
        if let Ok(cfg) = serde_json::from_reader::<_, SerialConfig>(file) {
            let trimmed = cfg.port_name.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }
    DEFAULT_SERIAL_PORT.to_string()
}

/// 시리얼 포트 경로를 파일("serial_config.json")에 영구 저장합니다.
pub fn save_serial_port_name(port_name: &str) -> std::io::Result<()> {
    let cfg = SerialConfig {
        port_name: port_name.trim().to_string(),
    };
    let file = std::fs::File::create(SERIAL_CONFIG_FILE)?;
    serde_json::to_writer_pretty(file, &cfg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(())
}

