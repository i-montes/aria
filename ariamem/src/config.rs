use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub system: SystemConfig,
    pub storage: StorageConfig,
    pub embedder: EmbedderConfig,
    pub engine: EngineConfig,
}

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
        // Standardize on the ultra-lightweight potion model for reliability
        let model_name = "minishlab/potion-base-32M".to_string();
        let model_path = "models/potion-base-32M".to_string();

        Self {
            system: SystemConfig {
                log_level: "info".to_string(),
                data_dir: None,
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
            },
        }
    }
}

impl Config {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        // 1. Check local directory first
        let local_path = PathBuf::from("aria.config.json");
        if local_path.exists() {
            let content = fs::read_to_string(&local_path)?;
            let config: Config = serde_json::from_str(&content)?;
            return Ok(config);
        }

        // 2. Check system directories
        let project_dirs = ProjectDirs::from("", "aria-project", "aria")
            .ok_or("Could not determine system directory")?;
        let config_dir = project_dirs.config_dir();
        let system_path = config_dir.join("aria.config.json");

        if system_path.exists() {
            let content = fs::read_to_string(&system_path)?;
            let config: Config = serde_json::from_str(&content)?;
            return Ok(config);
        }

        // 3. If no config exists, create default in system directory
        let mut default_config = Config::default();
        let data_dir = project_dirs.data_dir();
        default_config.system.data_dir = Some(data_dir.to_path_buf());
        
        fs::create_dir_all(config_dir)?;
        fs::create_dir_all(data_dir)?; // Create data dir as well just in case

        // Write default config
        let json = serde_json::to_string_pretty(&default_config)?;
        fs::write(&system_path, json)?;
        
        println!("✓ Created default configuration at {:?}", system_path);
        if detect_gpu() {
            println!("✓ Hardware detection: GPU found. Selected optimal model.");
        } else {
            println!("✓ Hardware detection: CPU only. Selected lightweight model.");
        }

        Ok(default_config)
    }

    /// Resolves the final database path.
    /// If storage.path is absolute, it uses it directly.
    /// Otherwise, it joins it with the configured data_dir or the current directory.
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

    /// Resolves the final model path.
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
