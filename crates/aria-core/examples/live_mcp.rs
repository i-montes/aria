//! Manual check against the real services (needs the network). Prints no secrets.
//!   TAVILY_KEY=... cargo run -p aria-core --example live_mcp -- <dir for mcp.json>
use aria_core::mcp::{self, NOTION_URL};
use aria_core::paths::AriaPaths;
use aria_core::validate::{check_key, KeyCheck, KeyKind};

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("a directory for the files"));
    let paths = AriaPaths { root: dir };
    paths.ensure_root().unwrap();

    println!("tavily with a fake key:  {:?}", check_key(KeyKind::Tavily, "tvly-not-a-real-key"));
    let key = std::env::var("TAVILY_KEY").unwrap_or_default();
    let verdict = check_key(KeyKind::Tavily, &key);
    println!("tavily with the real key: {verdict:?}");
    if verdict == KeyCheck::Valid {
        mcp::set_tavily_key(&paths, &key).unwrap();
        println!("wrote {} (connected: {:?})", paths.mcp().display(), mcp::connected(&paths));
    }

    match aria_core::oauth::mcp::start(NOTION_URL) {
        Ok(sign_in) => {
            let url = url::Url::parse(&sign_in.url).unwrap();
            let q: std::collections::HashMap<_, _> = url.query_pairs().into_owned().collect();
            println!("notion authorize: {}://{}{}", url.scheme(), url.host_str().unwrap(), url.path());
            println!(
                "  redirect_uri={} resource={} scope={:?} pkce={} state_len={}",
                q["redirect_uri"],
                q["resource"],
                q.get("scope"),
                q["code_challenge_method"],
                q["state"].len()
            );
        }
        Err(error) => println!("notion start failed: {error}"),
    }
}
