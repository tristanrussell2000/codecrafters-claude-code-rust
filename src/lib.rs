use serde_json::Value;

#[derive(thiserror::Error, Debug)]
pub enum AgentError {
    #[error("Missing expected field: {0}")]
    MissingPayloadField(String),

    #[error("LLM generated invalid JSON: {0}")]
    MalformedJson(#[from] serde_json::Error),

    #[error("Tool execution failed: {0}")]
    ToolExecution(String)
}

pub fn extract_tool_arguments(tool_call: &Value) -> Result<Value, Box<dyn std::error::Error>> {
    let args_str = tool_call
    .get("function")
    .and_then(|f| f.get("arguments"))
    .and_then(|a| a.as_str())
    .ok_or("Missing or invalid 'arguments' string in tool call")?;

    let arguments: Value = serde_json::from_str(args_str)?;

    Ok(arguments)
}