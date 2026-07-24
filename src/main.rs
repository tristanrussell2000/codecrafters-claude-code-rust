use async_openai::{Client, config::OpenAIConfig, types::chat::{ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage, ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionTools, CreateChatCompletionRequestArgs, FunctionObjectArgs}};
use clap::Parser;
use serde_json::{Value, json};
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

    let mut messages: Vec<ChatCompletionRequestMessage> = vec![
        ChatCompletionRequestUserMessageArgs::default().content(args.prompt.clone()).build()?.into()
    ];

    let tools: Vec<ChatCompletionTools> = vec![ ChatCompletionTools::Function(ChatCompletionTool { function: FunctionObjectArgs::default()
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

        // Append the model's response to the conversation. Only set fields that
        // are present — the builder strips `Option`, so it wants the inner value.
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

        // No tool calls means the model gave a final answer — print it and stop.
        let Some(tool_calls) = message.tool_calls else {
            if let Some(content) = message.content {
                println!("{content}");
            }
            break;
        };

        // Otherwise run each tool call and feed its result back as a tool message.
        for tool_call in tool_calls {
            let ChatCompletionMessageToolCalls::Function(call) = tool_call else {
                continue;
            };

            if call.function.name != "Read" {
                continue;
            }

            let arguments: Value = match serde_json::from_str(&call.function.arguments) {
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

            let result = match tokio::fs::read_to_string(file_path).await {
                Ok(contents) => contents,
                Err(e) => format!("Error reading {file_path}: {e}"),
            };

            let tool_message = ChatCompletionRequestToolMessageArgs::default()
                .content(result)
                .tool_call_id(call.id)
                .build()?;
            messages.push(tool_message.into());
        }
    }

    Ok(())
}
