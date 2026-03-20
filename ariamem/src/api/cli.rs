use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ariamem")]
#[command(about = "AriaMem - Hybrid Memory Engine for AI Agents")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Init {
        #[arg(long)]
        path: Option<String>,
    },
    Store {
        #[arg(short, long)]
        content: String,
        #[arg(short, long, default_value = "world")]
        memory_type: String,
    },
    Query {
        #[arg(short, long)]
        text: String,
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    List {
        #[arg(short, long)]
        memory_type: Option<String>,
    },
    Stats,
}

pub fn run() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Init { path } => {
            println!("Initializing AriaMem at: {:?}", path.unwrap_or_default());
        }
        Commands::Store { content, memory_type } => {
            println!("Storing: {} (type: {})", content, memory_type);
        }
        Commands::Query { text, limit } => {
            println!("Querying: {} (limit: {})", text, limit);
        }
        Commands::List { memory_type } => {
            println!("Listing memories of type: {:?}", memory_type);
        }
        Commands::Stats => {
            println!("AriaMem Stats");
        }
    }
}
