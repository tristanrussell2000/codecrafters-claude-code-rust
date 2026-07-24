#[derive(thiserror::Error, Debug)]
pub enum AgentError {
    #[error("Missing expected field: {0}")]
    MissingPayloadField(String),

    #[error("LLM generated invalid JSON: {0}")]
    MalformedJson(#[from] serde_json::Error),

    #[error("Tool execution failed: {0}")]
    ToolExecution(String)
}