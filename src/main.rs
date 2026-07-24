use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use codecrafters_claude_code::extract_tool_arguments;
use serde_json::{Value, json};
use tokio::{fs::File, io};
use std::{env, process};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let args = Args::parse();
    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);
    
    #[allow(unused_variables)]
    let response: Value = client
        .chat()
        .create_byot(json!({
            "messages": [
                {
                    "role": "user",
                    "content": args.prompt
                }
            ],
            "model": "anthropic/claude-haiku-4.5",
            "tools": [{
              "type": "function",
              "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                  "type": "object",
                  "properties": {
                    "file_path": {
                      "type": "string",
                      "description": "The path to the file to read"
                    }
                  },
                  "required": ["file_path"]
                }
              }
            }]
        }))
        .await?;

    // You can use print statements as follows for debugging, they'll be visible when running tests.
    eprintln!("Logs from your program will appear here!");
    
    if let Some(tool_calls) = response["choices"][0]["message"]["tool_calls"].as_array() {
        for tool_call in tool_calls {
            let Some(name) = tool_call.get("function").and_then(|f| f.get("name")).and_then(|f| f.as_str()) else {
                continue;
            };
            
            if name != "Read" {
                continue;
            }

            let arguments: Value = match extract_tool_arguments(tool_call) {
                Ok(args) => args,
                Err(e) => {
                    eprintln!("Failed to parse Read arguments: {e}");
                    continue;
                }
            };

            let Some(file_path) = arguments["file_path"].as_str() else {
                eprintln!("Failed to parse Read arguments - file path not found");
                continue;
            };

            let mut file = File::open(file_path).await?;

            let mut stdout = io::stdout();

            io::copy(&mut file, &mut stdout).await?;
        }
    }

    if let Some(content) = response["choices"][0]["message"]["content"].as_str() {
        println!("{}", content);
    }

    Ok(())
}
