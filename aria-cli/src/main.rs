use clap::{Parser, Subcommand};
use std::process::{Command, Stdio};
use std::env;
use std::path::PathBuf;
use std::fs;
use sysinfo::System;
use serde_json::{json, Value};
use serde::Deserialize;

#[derive(Parser)]
#[command(name = "aria")]
#[command(about = "Aria Global Orchestrator")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(trailing_var_arg = true)]
    goal: Vec<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start a background service (mem | core)
    Start {
        module: String,
    },
    /// Stop a running service
    Stop {
        module: String,
    },
    /// Show status of all services
    Status,
    /// Interact with the Memory module
    Mem {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

#[derive(Deserialize, Debug)]
struct AriaConfig {
    #[allow(dead_code)]
    workspace_root: String,
    database_path: String,
    memory_port: u16,
    core_port: u16,
    #[serde(flatten)]
    extra: serde_json::Value,
}

impl AriaConfig {
    fn load() -> Self {
        let config_path = PathBuf::from("aria.config.json");
        if config_path.exists() {
            if let Ok(content) = fs::read_to_string(config_path) {
                if let Ok(config) = serde_json::from_str(&content) {
                    return config;
                }
            }
        }
        // Fallback defaults
        Self {
            workspace_root: ".".into(),
            database_path: "aria_whiteboard.db".into(),
            memory_port: 8080,
            core_port: 3000,
        }
    }
}

fn get_module_path(module: &str) -> Option<PathBuf> {
    let mut exe_dir = env::current_exe().unwrap();
    exe_dir.pop();
    let bin_name = if cfg!(windows) { format!("aria{}.exe", module) } else { format!("aria{}", module) };
    let path = exe_dir.join(&bin_name);
    if path.exists() { Some(path) } else { Some(PathBuf::from(&bin_name)) }
}

async fn send_goal_to_core(goal: &str, port: u16) {
    let client = reqwest::Client::new();
    match client.post(format!("http://127.0.0.1:{}/goal", port))
        .json(&json!({ "goal": goal }))
        .send()
        .await 
    {
        Ok(res) => {
            if res.status().is_success() {
                println!("✓ Goal sent to ARIA Core: '{}'", goal);
                let json: serde_json::Value = res.json().await.unwrap();
                println!("Response: {}", json);
            } else {
                eprintln!("✗ Core returned error: {}", res.status());
            }
        }
        Err(_) => {
            eprintln!("✗ ARIA Core is not running on port {}. Start it with 'aria start core'", port);
        }
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let config = AriaConfig::load();

    match cli.command {
        Some(Commands::Start { module }) => {
            if module != "mem" && module != "core" {
                eprintln!("Unknown module: {}. Available: mem, core", module);
                return;
            }
            let bin = get_module_path(&module).unwrap();
            println!("Starting {} module in background...", module);
            
            let mut cmd = Command::new(bin);
            if module == "mem" {
                cmd.arg("serve").arg("--port").arg(config.memory_port.to_string()).arg("--rest-port").arg((config.memory_port + 1010).to_string());
            }
            
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
            }
            
            cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
               
            match cmd.spawn() {
                Ok(child) => println!("✓ Started module '{}' (PID: {})", module, child.id()),
                Err(e) => eprintln!("✗ Failed to start {}: {}", module, e),
            }
        }
        Some(Commands::Stop { module }) => {
            let target_name = if cfg!(windows) { format!("aria{}.exe", module) } else { format!("aria{}", module) };
            let mut sys = System::new_all();
            sys.refresh_all();
            let mut found = false;
            for (pid, process) in sys.processes() {
                if process.name().to_string_lossy() == target_name {
                    println!("Stopping {} (PID: {})...", target_name, pid);
                    process.kill();
                    found = true;
                }
            }
            if !found { println!("No running process found for module: {}", module); }
            else { println!("✓ Stopped."); }
        }
        Some(Commands::Status) => {
            println!("ARIA SYSTEM STATUS:");
            println!("-------------------");
            let mut sys = System::new_all();
            sys.refresh_all();
            let modules = vec!["mem", "core"];
            for module in modules {
                let target_name = if cfg!(windows) { format!("aria{}.exe", module) } else { format!("aria{}", module) };
                let mut is_running = false;
                let mut pids = Vec::new();
                for (pid, process) in sys.processes() {
                    if process.name().to_string_lossy() == target_name { is_running = true; pids.push(pid.to_string()); }
                }
                if is_running { println!("🟢 {} \t RUNNING (PID: {})", module.to_uppercase(), pids.join(", ")); }
                else { println!("🔴 {} \t STOPPED", module.to_uppercase()); }
            }
            println!("-------------------");
        }
        Some(Commands::Mem { args }) => {
            let bin = get_module_path("mem").unwrap();
            let status = Command::new(bin).args(&args).status().expect("Failed to execute ariamem");
            if !status.success() { std::process::exit(status.code().unwrap_or(1)); }
        }
        None => {
            if !cli.goal.is_empty() {
                let goal = cli.goal.join(" ");
                send_goal_to_core(&goal, config.core_port).await;
            } else {
                println!("ARIA Global Orchestrator. Usage: aria <goal> or aria <command>");
            }
        }
    }
}
