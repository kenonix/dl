pub mod cnn;
pub mod cnn_lstm;
pub mod lstm;
pub mod mlp;
pub mod modular_mlp;
pub mod tcn;

pub use cnn::EmgCnnModel;
pub use cnn_lstm::EmgCnnLstmModel;
pub use lstm::EmgLstmModel;
pub use mlp::{EmgMlpConfig, EmgMlpModel};
pub use modular_mlp::{
    EmgModularMlpConfig, EmgModularMlpModel, RealTimeRingBuffer, SpatioTemporalMlpBlock,
    SwiGLUChannelBlock, TemporalMixingBlock, WarmupCosineLrScheduler, MODULAR_MLP_WINDOW_SIZE,
};
pub use tcn::EmgTcnModel;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    Mlp,
    ModularMlp,
    Cnn,
    Lstm,
    CnnLstm,
    Tcn,
}

impl ModelArchitecture {
    pub fn file_name(&self) -> &'static str {
        match self {
            Self::Mlp => "emg_mlp_model",
            Self::ModularMlp => "emg_modular_mlp_model",
            Self::Cnn => "emg_cnn_model",
            Self::Lstm => "emg_lstm_model",
            Self::CnnLstm => "emg_cnnlstm_model",
            Self::Tcn => "emg_tcn_model",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Mlp => "MLP (Dense 5-Layer)",
            Self::ModularMlp => "Block-MLP (Temporal-SwiGLU Modular NN)",
            Self::Cnn => "1D-CNN (BatchNorm)",
            Self::Lstm => "LSTM",
            Self::CnnLstm => "CNN + LSTM (CRNN)",
            Self::Tcn => "1D-TCN (Temporal Convolutional Network)",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().replace('-', "_").as_str() {
            "mlp" | "dense" | "emg_mlp" => Some(Self::Mlp),
            "modular_mlp" | "block_mlp" | "swiglu_mlp" | "temporal_mlp" | "emg_modular_mlp" => {
                Some(Self::ModularMlp)
            }
            "cnn" | "1d_cnn" | "1dcnn" => Some(Self::Cnn),
            "lstm" => Some(Self::Lstm),
            "cnn_lstm" | "cnnlstm" | "crnn" => Some(Self::CnnLstm),
            "tcn" | "1d_tcn" | "1dtcn" => Some(Self::Tcn),
            _ => None,
        }
    }
}
