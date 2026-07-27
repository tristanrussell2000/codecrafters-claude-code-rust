use async_openai::{Client, config::OpenAIConfig, types::chat::{ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage, ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionTools, CreateChatCompletionRequestArgs, FunctionObjectArgs}};
use clap::Parser;
use serde_json::{Value, json};
use std::{env, process::{self, Command}};

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

    let mut messages: Vec<ChatCompletionRequestMessage> = vec![
        ChatCompletionRequestUserMessageArgs::default().content(args.prompt.clone()).build()?.into()
    ];

    let tools: Vec<ChatCompletionTools> = vec![ 
        ChatCompletionTools::Function(ChatCompletionTool { function: FunctionObjectArgs::default()
            .name("Read")
            .description("Read and return the contents of a file")
            .parameters(json!({
                "type": "object",
                "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to read"
                }
                },
                "required": ["file_path"]
            }))
            .build()?
            .into()
        }),
        ChatCompletionTools::Function(ChatCompletionTool {
            function: FunctionObjectArgs::default()
            .name("Write")
            .description("Write content to a file")
            .parameters(json!({
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                  "file_path": {
                    "type": "string",
                    "description": "The path of the file to write to"
                  },
                  "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                  }
                }
            }))
            .build()?
            .into()
        }),
        ChatCompletionTools::Function(ChatCompletionTool { 
            function: FunctionObjectArgs::default()
            .name("Bash")
            .description("Execute a shell command")
            .parameters(json!({
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute"
                    }
                }
            }))
            .build()?
            .into()
        })
    ];

    loop {
        let request = CreateChatCompletionRequestArgs::default()
            .model("anthropic/claude-haiku-4.5")
            .messages(messages.clone())
            .tools(tools.clone())
            .build()?;

        let response = client.chat().create(request).await?;

        let Some(choice) = response.choices.into_iter().next() else {
            break;
        };
        let message = choice.message;

        let mut assistant = ChatCompletionRequestAssistantMessageArgs::default();
        if let Some(content) = message.content.clone() {
            assistant.content(content);
        }
        if let Some(refusal) = message.refusal {
            assistant.refusal(refusal);
        }
        if let Some(tool_calls) = message.tool_calls.clone() {
            assistant.tool_calls(tool_calls);
        }
        messages.push(assistant.build()?.into());

        let Some(tool_calls) = message.tool_calls else {
            if let Some(content) = message.content {
                println!("{content}");
            }
            break;
        };

        for tool_call in tool_calls {
            let ChatCompletionMessageToolCalls::Function(call) = tool_call else {
                continue;
            };

            let result = run_tool(&call).await.unwrap_or_else(|err| err);

            let tool_message = ChatCompletionRequestToolMessageArgs::default()
                .content(result)
                .tool_call_id(call.id)
                .build()?;
            messages.push(tool_message.into());
        }
    }

    Ok(())
}

async fn run_tool(call: &ChatCompletionMessageToolCall) -> Result<String, String> {
    let arguments: Value = serde_json::from_str(&call.function.arguments)
        .map_err(|e| format!("Failed to parse tool arguments as JSON: {e}"))?;

    match call.function.name.as_str() {
        "Read" => {
            let file_path = arguments["file_path"]
                .as_str()
                .ok_or("Missing required argument 'file_path'")?;
            tokio::fs::read_to_string(file_path)
                .await
                .map_err(|e| format!("Error reading {file_path}: {e}"))
        }
        "Write" => {
            let file_path = arguments["file_path"]
                .as_str()
                .ok_or("Missing required argument 'file_path'")?;
            let content = arguments["content"]
                .as_str()
                .ok_or("Missing required argument 'content'")?;
            tokio::fs::write(file_path, content)
                .await
                .map_err(|e| format!("Error writing {file_path}: {e}"))?;
            Ok(format!("Wrote {} bytes to {file_path}", content.len()))
        }
        "Bash" => {
            let command = arguments["command"]
            .as_str()
            .ok_or("Missing required argument 'command'")?;

            let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()
            .map_err(|e| format!("Error using bash command: {e}"))?;

            let output_str = String::from_utf8(output.stdout).map_err(|e| format!("Command succeeded but failed to parse stdout as a string: {e}"))?;
            Ok(output_str)
        }
        other => Err(format!("Unknown tool '{other}'")),
    }
}
