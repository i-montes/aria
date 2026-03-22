use ariamem::api::cli;

#[tokio::main]
async fn main() {
    cli::run().await;
}
