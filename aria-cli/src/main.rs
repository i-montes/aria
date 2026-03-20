use clap::{Parser, Subcommand};
use std::process::{Command, Stdio};
use std::env;
use std::path::PathBuf;
use sysinfo::System;

#[derive(Parser)]
#[command(name = "aria")]
#[command(about = "Aria Global Orchestrator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start a background service
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

fn get_module_path(module: &str) -> Option<PathBuf> {
    let mut exe_dir = env::current_exe().unwrap();
    exe_dir.pop(); // Remove aria.exe to get directory
    
    // In dev mode (target/debug or release), the module is next to us
    let bin_name = if cfg!(windows) {
        format!("aria{}.exe", module)
    } else {
        format!("aria{}", module)
    };
    
    let path = exe_dir.join(&bin_name);
    if path.exists() {
        Some(path)
    } else {
        // Fallback for global install (if it's in PATH)
        Some(PathBuf::from(&bin_name))
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Start { module } => {
            if module != "mem" {
                eprintln!("Unknown module: {}", module);
                return;
            }
            let bin = get_module_path(&module).unwrap();
            
            println!("Starting {} module in background...", module);
            
            let mut cmd = Command::new(bin);
            cmd.arg("serve");
            cmd.arg("--port");
            cmd.arg("8080");
            
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
            }
            
            cmd.stdin(Stdio::null())
               .stdout(Stdio::null())
               .stderr(Stdio::null());
               
            match cmd.spawn() {
                Ok(child) => println!("✓ Started module '{}' (PID: {})", module, child.id()),
                Err(e) => eprintln!("✗ Failed to start {}: {}", module, e),
            }
        }
        Commands::Stop { module } => {
            let target_name = if cfg!(windows) {
                format!("aria{}.exe", module)
            } else {
                format!("aria{}", module)
            };
            
            let mut sys = System::new_all();
            sys.refresh_all();
            
            let mut found = false;
            for (pid, process) in sys.processes() {
                if process.name() == target_name {
                    println!("Stopping {} (PID: {})...", target_name, pid);
                    process.kill();
                    found = true;
                }
            }
            
            if !found {
                println!("No running process found for module: {}", module);
            } else {
                println!("✓ Stopped.");
            }
        }
        Commands::Status => {
            println!("ARIA SYSTEM STATUS:");
            println!("-------------------");
            
            let mut sys = System::new_all();
            sys.refresh_all();
            
            let modules = vec!["mem"];
            
            for module in modules {
                let target_name = if cfg!(windows) {
                    format!("aria{}.exe", module)
                } else {
                    format!("aria{}", module)
                };
                
                let mut is_running = false;
                let mut module_pids = Vec::new();
                
                for (pid, process) in sys.processes() {
                    if process.name() == target_name {
                        is_running = true;
                        module_pids.push(pid.to_string());
                    }
                }
                
                if is_running {
                    println!("🟢 {} \t RUNNING (PID: {})", module.to_uppercase(), module_pids.join(", "));
                } else {
                    println!("🔴 {} \t STOPPED", module.to_uppercase());
                }
            }
            println!("-------------------");
        }
        Commands::Mem { args } => {
            let bin = get_module_path("mem").unwrap();
            let status = Command::new(bin)
                .args(&args)
                .status()
                .expect("Failed to execute ariamem");
                
            if !status.success() {
                std::process::exit(status.code().unwrap_or(1));
            }
        }
    }
}