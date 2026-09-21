pub mod cnn;
pub mod lstm;
pub mod cnn_lstm;

pub use cnn::EmgCnnModel;
pub use lstm::EmgLstmModel;
pub use cnn_lstm::EmgCnnLstmModel;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    Cnn,
    Lstm,
    CnnLstm,
}

impl ModelArchitecture {
    pub fn file_name(&self) -> &'static str {
        match self {
            Self::Cnn => "emg_cnn_model",
            Self::Lstm => "emg_lstm_model",
            Self::CnnLstm => "emg_cnnlstm_model",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Cnn => "1D-CNN",
            Self::Lstm => "LSTM",
            Self::CnnLstm => "CNN + LSTM (CRNN)",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().replace('-', "_").as_str() {
            "cnn" | "1d_cnn" | "1dcnn" => Some(Self::Cnn),
            "lstm" => Some(Self::Lstm),
            "cnn_lstm" | "cnnlstm" | "crnn" => Some(Self::CnnLstm),
            _ => None,
        }
    }
}
