use axum::{
    body::Body,
    extract::{Multipart, Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use burn_wgpu::WgpuDevice;
use emg_ml_pipeline::{
    train_model, ModelArchitecture, TrainingConfig, TrainingProgress,
};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path as StdPath, PathBuf};
use std::sync::{Arc, RwLock};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerTrainingState {
    pub status: String, // "idle", "training", "completed", "failed"
    pub epoch: usize,
    pub total_epochs: usize,
    pub loss: f32,
    pub model_type: String,
    pub model_file: Option<String>,
    pub error: Option<String>,
}

impl Default for ServerTrainingState {
    fn default() -> Self {
        Self {
            status: "idle".to_string(),
            epoch: 0,
            total_epochs: 0,
            loss: 0.0,
            model_type: String::new(),
            model_file: None,
            error: None,
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub state: Arc<RwLock<ServerTrainingState>>,
    pub upload_dir: PathBuf,
    pub model_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
pub struct TrainRequest {
    pub dataset: String,
    pub model_type: String, // "cnn", "lstm", "cnn_lstm"
    pub epochs: Option<usize>,
    pub lr: Option<f64>,
    pub batch_size: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ModelFileInfo {
    pub name: String,
    pub size_bytes: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let upload_dir = PathBuf::from("uploads");
    let model_dir = PathBuf::from(".");
    fs::create_dir_all(&upload_dir)?;

    let app_state = AppState {
        state: Arc::new(RwLock::new(ServerTrainingState::default())),
        upload_dir,
        model_dir,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/upload", post(upload_dataset))
        .route("/api/train", post(start_training))
        .route("/api/status", get(get_status))
        .route("/api/models", get(list_models))
        .route("/api/download/:name", get(download_model))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(app_state);

    let addr = "0.0.0.0:3000";
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("==================================================");
    println!(" 🚀 순수 Rust EMG 딥러닝 학습 & 모델 서버 가동 중");
    println!("    - 주소: http://{}", addr);
    println!("    - 엔드포인트:");
    println!("      * POST /api/upload         : CSV 데이터셋 업로드");
    println!("      * POST /api/train          : 1D-TCN / 1D-CNN / LSTM / CNN+LSTM 원격 학습");
    println!("      * GET  /api/status         : 학습 진행률 및 상태");
    println!("      * GET  /api/models         : 학습된 모델 목록");
    println!("      * GET  /api/download/:name : .mpk 모델 바이너리 다운로드");
    println!("==================================================");

    axum::serve(listener, app).await?;
    Ok(())
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "emg_training_server"
    }))
}

/// CSV 데이터셋 파일 업로드 핸들러
async fn upload_dataset(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut saved_filename = None;
    let mut saved_size = 0usize;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("멀티파트 파싱 실패: {}", e)))?
    {
        let filename = field
            .file_name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("dataset_{}.csv", chrono::Local::now().format("%Y%m%d_%H%M%S")));

        // 보안: 파일명에 상위 디렉터리 경로 방지
        let clean_filename = StdPath::new(&filename)
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| (StatusCode::BAD_REQUEST, "유효하지 않은 파일명입니다.".into()))?;

        let save_path = state.upload_dir.join(clean_filename);
        let data = field
            .bytes()
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("데이터 수신 에러: {}", e)))?;

        saved_size = data.len();
        let mut file = File::create(&save_path)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("파일 생성 에러: {}", e)))?;
        file.write_all(&data)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("파일 쓰기 에러: {}", e)))?;

        saved_filename = Some(clean_filename.to_string());
    }

    match saved_filename {
        Some(name) => Ok(Json(serde_json::json!({
            "status": "ok",
            "filename": name,
            "size": saved_size,
            "message": "데이터셋이 성공적으로 업로드되었습니다."
        }))),
        None => Err((StatusCode::BAD_REQUEST, "업로드된 파일이 없습니다.".into())),
    }
}

/// 모델 학습 요청 핸들러 (비동기 백그라운드 워커 스레드)
async fn start_training(
    State(state): State<AppState>,
    Json(payload): Json<TrainRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    {
        let current_state = state.state.read().unwrap();
        if current_state.status == "training" {
            return Err((
                StatusCode::CONFLICT,
                "이미 다른 모델이 학습 중입니다. 완료 후 다시 시도해주세요.".into(),
            ));
        }
    }

    let model_arch = ModelArchitecture::from_str(&payload.model_type).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            format!("지원하지 않는 모델 유형입니다: {}", payload.model_type),
        )
    })?;

    // 업로드 폴더 또는 현재 폴더에서 데이터셋 파일 탐색
    let mut resolved_dataset = state.upload_dir.join(&payload.dataset);
    if !resolved_dataset.exists() {
        resolved_dataset = PathBuf::from(&payload.dataset);
        if !resolved_dataset.exists() {
            return Err((
                StatusCode::NOT_FOUND,
                format!("데이터셋 파일을 찾을 수 없습니다: {}", payload.dataset),
            ));
        }
    }

    let config = TrainingConfig {
        dataset_path: resolved_dataset.to_string_lossy().to_string(),
        model_arch,
        epochs: payload.epochs.unwrap_or(150),
        learning_rate: payload.lr.unwrap_or(1e-3),
        batch_size: payload.batch_size.unwrap_or(32),
    };

    // 상태를 "training"으로 초기화
    {
        let mut s = state.state.write().unwrap();
        s.status = "training".to_string();
        s.epoch = 0;
        s.total_epochs = config.epochs;
        s.loss = 0.0;
        s.model_type = model_arch.display_name().to_string();
        s.model_file = None;
        s.error = None;
    }

    let state_clone = Arc::clone(&state.state);

    // 무거운 학습 작업은 블로킹 스레드풀에서 실행
    tokio::task::spawn_blocking(move || {
        let device = WgpuDevice::default();
        let training_res = train_model(&device, &config, |prog: TrainingProgress| {
            if let Ok(mut s) = state_clone.write() {
                s.epoch = prog.epoch;
                s.total_epochs = prog.total_epochs;
                s.loss = prog.loss;
                s.status = prog.status;
                s.model_type = prog.model_type;
                s.model_file = prog.model_file;
            }
        });

        match training_res {
            Ok(model_path) => {
                println!("✔ [서버] 모델 학습 및 저장 완료: {}", model_path);
                if let Ok(mut s) = state_clone.write() {
                    s.status = "completed".to_string();
                    s.model_file = Some(model_path);
                }
            }
            Err(e) => {
                eprintln!("❌ [서버] 모델 학습 실패: {}", e);
                if let Ok(mut s) = state_clone.write() {
                    s.status = "failed".to_string();
                    s.error = Some(e.to_string());
                }
            }
        }
    });

    Ok(Json(serde_json::json!({
        "status": "started",
        "message": format!("{} 모델 학습이 시작되었습니다.", model_arch.display_name()),
        "model_type": model_arch.display_name(),
        "dataset": payload.dataset
    })))
}

/// 학습 진행 상태 조회
async fn get_status(State(state): State<AppState>) -> impl IntoResponse {
    let s = state.state.read().unwrap();
    Json(s.clone())
}

/// 다운로드 가능한 모델 파일 목록 조회
async fn list_models(State(state): State<AppState>) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut models = Vec::new();

    let search_dir = &state.model_dir;
    let read_dir = fs::read_dir(search_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("디렉터리 조회 실패: {}", e),
        )
    })?;

    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("mpk") {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let metadata = entry.metadata().ok();
                let size = metadata.map(|m| m.len()).unwrap_or(0);
                models.push(ModelFileInfo {
                    name: name.to_string(),
                    size_bytes: size,
                });
            }
        }
    }

    Ok(Json(models))
}

/// 모델 바이너리 파일 다운로드 핸들러
async fn download_model(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // 보안 검사
    if model_name.contains('/') || model_name.contains('\\') || model_name.contains("..") {
        return Err((StatusCode::BAD_REQUEST, "잘못된 파일명입니다.".into()));
    }

    let model_path = state.model_dir.join(&model_name);
    if !model_path.exists() {
        return Err((StatusCode::NOT_FOUND, "해당 모델 파일을 찾을 수 없습니다.".into()));
    }

    let file_bytes = tokio::fs::read(&model_path).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("파일 읽기 실패: {}", e),
        )
    })?;

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        "application/octet-stream".parse().unwrap(),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        format!("attachment; filename=\"{}\"", model_name)
            .parse()
            .unwrap(),
    );

    Ok((headers, Body::from(file_bytes)))
}
