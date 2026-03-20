pub struct RestApi {
    port: u16,
}

impl RestApi {
    pub fn new(port: u16) -> Self {
        Self { port }
    }

    pub fn start(&self) {
        println!("Starting REST API on port {}", self.port);
    }
}
