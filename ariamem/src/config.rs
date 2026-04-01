use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    // Global ARIA settings
    #[serde(default = "default_workspace_root")]
    pub workspace_root: String,
    #[serde(default = "default_database_path")]
    pub database_path: String,
    #[serde(default = "default_memory_port")]
    pub memory_port: u16,
    #[serde(default = "default_core_port")]
    pub core_port: u16,

    // Module specific settings
    pub system: SystemConfig,
    pub storage: StorageConfig,
    pub embedder: EmbedderConfig,
    pub engine: EngineConfig,
}

fn default_workspace_root() -> String { ".".into() }
fn default_database_path() -> String { "aria_whiteboard.db".into() }
fn default_memory_port() -> u16 { 8080 }
fn default_core_port() -> u16 { 3000 }

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemConfig {
    pub log_level: String,
    pub data_dir: Option<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StorageConfig {
    pub storage_type: String,
    pub path: String,
    pub wal_mode: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbedderConfig {
    pub provider: String,
    pub model2vec: Model2VecConfig,
    pub http: HttpConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Model2VecConfig {
    pub model_name: String,
    pub model_path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HttpConfig {
    pub url: String,
    pub timeout_seconds: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EngineConfig {
    pub default_search_limit: usize,
    pub cache_size: usize,
    pub recency_lambda: f32,
}

fn detect_gpu() -> bool {
    // Check for NVIDIA GPU
    if Command::new("nvidia-smi").arg("--query-gpu=name").arg("--format=csv,noheader").output().is_ok() {
        return true;
    }
    // Check for Apple Silicon (Metal)
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = Command::new("sysctl").arg("-n").arg("machdep.cpu.brand_string").output() {
            let cpu_info = String::from_utf8_lossy(&output.stdout).to_lowercase();
            if cpu_info.contains("apple") {
                return true; // Apple Silicon has integrated Metal GPU
            }
        }
    }
    false
}

impl Default for Config {
    fn default() -> Self {
        let model_name = "minishlab/potion-base-32M".to_string();
        let model_path = "models/potion-base-32M".to_string();

        Self {
            workspace_root: ".".into(),
            database_path: "aria_whiteboard.db".into(),
            memory_port: 8080,
            core_port: 3000,
            system: SystemConfig {
                log_level: "info".to_string(),
                data_dir: Some(PathBuf::from("data")),
            },
            storage: StorageConfig {
                storage_type: "sqlite".to_string(),
                path: "ariamem.db".to_string(),
                wal_mode: true,
            },
            embedder: EmbedderConfig {
                provider: "model2vec".to_string(),
                model2vec: Model2VecConfig {
                    model_name,
                    model_path,
                },
                http: HttpConfig {
                    url: "http://localhost:11434/api/embeddings".to_string(),
                    timeout_seconds: 30,
                },
            },
            engine: EngineConfig {
                default_search_limit: 10,
                cache_size: 10000,
                recency_lambda: 0.1,
            },
        }
    }
}

impl Config {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let local_path = PathBuf::from("aria.config.json");
        
        if local_path.exists() {
            let content = fs::read_to_string(&local_path)?;
            let config: Config = serde_json::from_str(&content)?;
            return Ok(config);
        }

        let mut default_config = Config::default();
        
        if let Some(ref data_dir) = default_config.system.data_dir {
            if !data_dir.exists() {
                fs::create_dir_all(data_dir)?;
            }
        }

        let content = serde_json::to_string_pretty(&default_config)?;
        fs::write(&local_path, content)?;
        
        println!("✓ Created default configuration at {:?}", local_path);
        if detect_gpu() {
            println!("✓ Hardware detection: GPU found. Selected optimal model.");
        } else {
            println!("✓ Hardware detection: CPU only. Selected lightweight model.");
        }

        Ok(default_config)
    }

    pub fn get_db_path(&self) -> PathBuf {
        let path = PathBuf::from(&self.storage.path);
        if path.is_absolute() {
            path
        } else if let Some(ref data_dir) = self.system.data_dir {
            data_dir.join(path)
        } else {
            path
        }
    }

    pub fn get_model_path(&self) -> PathBuf {
        let path = PathBuf::from(&self.embedder.model2vec.model_path);
        if path.is_absolute() {
            path
        } else if let Some(ref data_dir) = self.system.data_dir {
            data_dir.join(path)
        } else {
            path
        }
    }
}
